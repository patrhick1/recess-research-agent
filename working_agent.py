import os
import re
import asyncio
import tiktoken
import datetime
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from typing import Annotated, List, Optional, Literal, Union, Dict, Any, get_type_hints, get_origin, get_args
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from utils import log_message
import operator
import json

# LangChain imports
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

# Visualization imports
from IPython.display import display, Image
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#############################################
# DATA MODELS
#############################################

# Configuration data Model
class Config(BaseModel):
    research_type: str = Field(
        description="Type of research report (Academic, Technical, Business, or Other)"
    )
    target_audience: str = Field(
        description="Who the report is intended for"
    )
    writing_style: str = Field(
        description="The tone and style of writing"
    )


# Evidence point models
class EvidencePoint(BaseModel):
    fact: str = Field(description="A specific fact, statistic, or example")
    source: str = Field(description="Source of this evidence")
    relevance: str = Field(description="How this evidence relates to the section topic")
    subsection: str = Field(description="Suggested subsection where this evidence belongs")


class EvidencePoints(BaseModel):
    evidence_points: List[EvidencePoint] = Field(
        description="All the Evidence Points extracted from search results.",
    )


# Paragraph structure
class SubPoint(BaseModel):
    content: str = Field(description="Content for this specific point")
    sources: List[str] = Field(description="Sources supporting this point")


class Paragraph(BaseModel):
    main_idea: str = Field(description="The central idea of this paragraph")
    points: List[SubPoint] = Field(description="Supporting points for this paragraph")
    synthesized_content: str = Field(description="Final paragraph text synthesized from points")


class SubSection(BaseModel):
    title: str = Field(description="Title of this subsection")
    paragraphs: List[Paragraph] = Field(description="Paragraphs in this subsection")
    synthesized_content: str = Field(description="Final subsection text synthesized from paragraphs")


class Section(BaseModel):
    name: str = Field(description="Name for a particular section of the report")
    description: str = Field(description="Brief overview of the main topics and concepts")
    research: bool = Field(description="Whether to perform web search for this section")
    subsections: List[SubSection] = Field(description="Subsections within this section")
    content: str = Field(description="The final content for this section")
    search_docs: Optional[List[dict]] = Field(
        default_factory=list,
        description="Raw search docs or references used by this section"
    )
    evidence_points: Optional[List[EvidencePoint]] = Field(
        default_factory=list,
        description="Evidence points extracted for this section"
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="All the Sections of the overall report.",
    )


# Search query models
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of web search queries.",
    )


class CompanyBrand(BaseModel):
    brand_name: str = Field(description="Name of the brand")
    category: str = Field(description="Main product category (e.g., yogurt, snacks, beverages)")
    subcategory: Optional[str] = Field(description="More specific product type if available")
    description: Optional[str] = Field(description="Brief description of the brand's products")

class CompanyBrands(BaseModel):
    brands: List[CompanyBrand] = Field(description="List of brands owned by the company")

# State models for graph
class ReportStateInput(TypedDict):
    company_name: str  # Company name
    time_period: str  # Time period
    topic: str  # Combined topic for search purposes
    config: dict


class ReportStateOutput(TypedDict):
    final_report: str  # Final report


class ReportState(TypedDict):
    company_name: str  # Company name
    time_period: str  # Time period
    topic: str
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[list[Section], operator.add]
    report_sections_from_research: str
    final_report: str
    filename: str
    config: dict
    company_brands: list[dict]  # List of company brands and their categories


class SectionState(TypedDict):
    section: Section  # Report section
    search_queries: list[SearchQuery]  # List of search queries
    source_str: str  # String of formatted source content from web search
    report_sections_from_research: str  # completed sections to write final sections
    completed_sections: list[Section]  # Final key in outer state for Send() API
    evidence_points: list[EvidencePoint]  # Evidence points for this section
    company_name: str  # Company name passed from ReportState
    time_period: str  # Time period passed from ReportState
    company_brands: list[dict]  # List of company brands and their categories passed from ReportState


class SectionOutputState(TypedDict):
    completed_sections: list[Section]


# Helper dataclass for SearchQuery
@dataclass
class SearchQuery:
    search_query: str
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#############################################
# UTILITIES
#############################################

# Build references from the search docs stored in each section
def build_references(search_docs: List[dict]) -> str:
    """
    Build references with full URLs and clean formatting.
    Format: Author/Organization (Year). Title. URL
    """
    current_year = datetime.datetime.now().year
    unique_sources = {}

    # Process documents
    for doc in search_docs:
        if isinstance(doc, dict):
            if "results" in doc:
                for source in doc.get("results", []):
                    if url := source.get("url"):
                        unique_sources.setdefault(url, source)
            elif url := doc.get("url"):
                unique_sources.setdefault(url, doc)

    citations = []
    for source in unique_sources.values():
        # Clean title
        title = source.get("title", "Untitled Document").strip()
        title = re.sub(r'\s+', ' ', title)  # Collapse whitespace
        title = re.sub(r'\s*-\s*[^\-]*$', '', title)  # Remove trailing hyphens only at end
        
        # Extract components
        url = source.get("url", "")
        content = source.get("content", "")
        
        # Year extraction (supports 1900-2099)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', content)
        year = year_match.group(1) if year_match else str(current_year)

        # Publisher extraction with www handling
        domain_match = re.findall(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match[0].lower().replace("www.", "", 1)
            publisher = domain.split('.')[0].title()
        else:
            publisher = "Online Source"

        # Format citation
        citation = f"{publisher}. ({year}). {title}. {url}"
        citations.append(citation)

    if not citations:
        return "## References\n\nNo references available."

    # Deduplicate and sort
    return "## References\n\n" + "\n\n".join(
        f"- {cite}" for cite in sorted(
            set(citations),
            key=lambda x: x.split(". ")[0].lower()
        )
    )



# Initialize API clients
tavily_search = TavilySearchAPIWrapper()


def get_llm(llm_model: str):
    if llm_model == "o3-mini":
        llm = ChatOpenAI(model_name="o3-mini", temperature=None, openai_api_key=OPENAI_KEY, reasoning_effort="medium")
    elif llm_model == "gpt-4o":
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_KEY)
    elif llm_model == "gemini-2.0-flash":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
    return llm

llm = get_llm("o3-mini")

def generate_prompt(placeholders: dict, prompt_file: str) -> str:
    """
    Replaces placeholders in the prompt template with their respective values.
    Args:
        placeholders (dict): keys are placeholders, values are replacements
        prompt_file (str): path to the prompt file
    Returns:
        str: The modified prompt with placeholders replaced.
    """
    with open(prompt_file, 'r', encoding='utf-8') as file:
        prompt = file.read()

    for key, value in placeholders.items():
        prompt = prompt.replace(f'{{{key}}}', str(value))
    return prompt


def format_sections(sections: list[Section]) -> str:
    """Format a list of report sections into a single text string."""
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
        {'='*60}
        Section {idx}: {section.name}
        {'='*60}
        Description:
        {section.description}
        Requires Research:
        {section.research}
        Content:
        {section.content if section.content else '[Not yet written]'}
        """
    return formatted_str


def generate_report_filename(report_content: str) -> str:
    """Generate a filename based on report content using LLM."""
    title_prompt = """Analyze this report content and generate a concise filename in snake_case format. 
    Follow these rules:
    1. Use only lowercase letters, numbers, and underscores
    2. Maximum 5 words
    3. Reflect main topic from first section
    4. No special characters or spaces
    
    Content: {content}"""

    response = llm.invoke([
        SystemMessage(content=title_prompt.format(content=report_content[:2000])),
        HumanMessage(content="Generate filename following the rules:")
    ])

    # Clean up any extra quotes or spaces
    return response.content.strip().replace('"', '').replace("'", "").replace(" ", "_") + ".md"


#############################################
# WEB SEARCH FUNCTIONS
#############################################

async def run_search_queries(
    search_queries: List[Union[str, SearchQuery]],
    num_results: int = 5,
    include_raw_content: bool = False
) -> List[Dict]:
    search_tasks = []
    for query in search_queries:
        # Handle both string and SearchQuery objects
        query_str = query.search_query if isinstance(query, SearchQuery) else str(query)
        try:
            # get results from tavily async
            search_tasks.append(
                tavily_search.raw_results_async(
                    query=query_str,
                    max_results=num_results,
                    search_depth='advanced',
                    include_answer=False,
                    include_raw_content=include_raw_content
                )
            )
        except Exception as e:
            print(f"Error creating search task for query '{query_str}': {e}")
            continue
    # Execute all searches concurrently
    try:
        if not search_tasks:
            return []
        search_docs = await asyncio.gather(*search_tasks, return_exceptions=True)
        valid_results = [
            doc for doc in search_docs
            if not isinstance(doc, Exception)
        ]
        return valid_results
    except Exception as e:
        print(f"Error during search queries: {e}")
        return []


def format_search_query_results(
    search_response: Union[Dict[str, Any], List[Any]],
    max_tokens: int = 2000,
    include_raw_content: bool = False
) -> str:
    encoding = tiktoken.encoding_for_model("gpt-4")
    sources_list = []

    # Handle different response formats
    if isinstance(search_response, dict):
        if 'results' in search_response:
            sources_list.extend(search_response['results'])
        else:
            sources_list.append(search_response)
    elif isinstance(search_response, list):
        for response in search_response:
            if isinstance(response, dict):
                if 'results' in response:
                    sources_list.extend(response['results'])
                else:
                    sources_list.append(response)
            elif isinstance(response, list):
                sources_list.extend(response)

    if not sources_list:
        return "No search results found."

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if isinstance(source, dict) and 'url' in source:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source

    formatted_text = "Content from web search:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source.get('title', 'Untitled')}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source.get('content', 'No content available')}\n===\n"

        if include_raw_content:
            raw_content = source.get("raw_content", "")
            if raw_content:
                tokens = encoding.encode(raw_content)
                truncated_tokens = tokens[:max_tokens]
                truncated_content = encoding.decode(truncated_tokens)
                formatted_text += f"Raw Content: {truncated_content}\n\n"

    return formatted_text.strip()


#############################################
# REPORT PLANNING FUNCTIONS
#############################################

async def generate_report_plan(state: ReportState):
    """
    Parse the report structure from the predefined file instead of generating it.
    The structure file contains sections and subsections organized in a specific format.
    """
    log_message('--- Reading Report Structure from File ---')
    company_name = state["company_name"]
    time_period = state["time_period"]
    

    log_message(f"DEBUG: Config type: {type(state.get('config'))}")
    log_message(f"DEBUG: Config content: {state.get('config')}")
    
    # Define the path to the report structure file
    structure_file = os.path.join("prompts", "report_structure.txt")
    
    try:
        # Check if the file exists
        if not os.path.exists(structure_file):
            raise FileNotFoundError(f"Report structure file not found: {structure_file}")
        
        # Read the structure file
        with open(structure_file, 'r', encoding='utf-8') as f:
            structure_content = f.read()
        
        # Parse the structure content to extract sections and subsections
        sections = []
        current_section = None
        current_subsections = []
        
        lines = structure_content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Process section header
            if line.startswith('# Section:'):
                # If we already have a section in progress, add it to the list
                if current_section:
                    # Create subsection objects
                    subsections = []
                    for subsec_title in current_subsections:
                        subsections.append(
                            SubSection(
                                title=subsec_title,
                                paragraphs=[],
                                synthesized_content=""
                            )
                        )
                    
                    # Update the section with the subsections
                    current_section.subsections = subsections
                    sections.append(current_section)
                
                # Start a new section
                section_name = line.replace('# Section:', '').strip()
                current_section = Section(
                    name=section_name,
                    description="",  # Will be updated below
                    research=False,  # Default, will be updated below
                    subsections=[],  # Will be populated later
                    content="",
                    search_docs=[]
                )
                current_subsections = []
                
            # Process research flag
            elif line.startswith('Research:'):
                if current_section:
                    research_value = line.replace('Research:', '').strip().lower()
                    current_section.research = research_value == 'yes'
                    
            # Process description
            elif line.startswith('Description:'):
                if current_section:
                    current_section.description = line.replace('Description:', '').strip()
                    
            # Process subsection
            elif line.startswith('## Subsection:'):
                if current_section:
                    subsection_title = line.replace('## Subsection:', '').strip()
                    current_subsections.append(subsection_title)
        
        # Add the final section if there is one
        if current_section:
            # Create subsection objects for the final section
            subsections = []
            for subsec_title in current_subsections:
                subsections.append(
                    SubSection(
                        title=subsec_title,
                        paragraphs=[],
                        synthesized_content=""
                    )
                )
                
            # Update the section with the subsections
            current_section.subsections = subsections
            sections.append(current_section)
        
        log_message(f'--- Read {len(sections)} Sections from Structure File ---')
        
        # Replace {Company Name} placeholder in descriptions with the actual company name
        # and add time period context
        for section in sections:
            section.description = section.description.replace("{Company Name}", company_name)
            
            # Add time period context to the description if not already present
            if time_period and time_period.lower() not in section.description.lower():
                section.description += f" Analysis covers the {time_period} period."
            
        return {"sections": sections, "company_name": company_name, "time_period": time_period}
        
    except Exception as e:
        print(f"Error in generate_report_plan: {e}")
        return {"sections": [], "company_name": company_name, "time_period": time_period}


async def discover_company_brands(state: ReportState):
    """
    Search for brands and product categories belonging to the company using Tavily.
    Updates state with a list of brands and their respective categories.
    """
    log_message(f"🔍 Discovering brands for {state['company_name']}...")
    
    # Create search query specifically to find brands owned by the company
    brand_search_query = f"List of all brands owned by {state['company_name']} in the United States market with their product categories"
    
    # Use existing search function to perform the search
    search_results = await run_search_queries(
        [SearchQuery(search_query=brand_search_query)],
        num_results=10,  # Increase number of results to get more comprehensive data
        include_raw_content=True
    )
    
    # Format the search results
    search_content = format_search_query_results(search_results, max_tokens=4000, include_raw_content=True)
    
    # Use structured output with Pydantic model
    structured_llm = llm.with_structured_output(CompanyBrands)
    
    prompt = f"""
    Based on the following search results about {state['company_name']}'s brand portfolio, 
    extract a comprehensive list of all brands owned by {state['company_name']} and their 
    respective product categories. Focus on brands currently active in the United States market.
    
    Search results:
    {search_content}
    
    Return a structured list of brands with their categories, subcategories, and descriptions.
    Include as many brands as you can identify from the search results.
    """
    
    try:
        # Get structured response
        brands_data = await structured_llm.ainvoke(prompt)
        
        # Convert to dict format and update state
        state["company_brands"] = [brand.model_dump() for brand in brands_data.brands]

        log_message(f"✅ Discovered {len(state['company_brands'])} brands for {state['company_name']}")
    except Exception as e:
        # Log error and use empty list as fallback
        log_message(f"⚠️ Error extracting brand information: {str(e)}. Using empty list as fallback.")
        state["company_brands"] = []
    
    # Ensure the function returns the complete state
    return state


def generate_queries(state: SectionState):
    """Generate search queries for a specific report section."""
    section = state["section"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    company_brands = state.get("company_brands", [])  # Get company brands if available
    
    log_message(f'--- Generating Search Queries for Section: {section.name} ---')
    log_message(f'LOGGING: Generating queries for {company_name}, {time_period}, section {section.name}')
    
    # CRITICAL FIX: Check if company_name is empty and use fallback from section.description if available
    if not company_name.strip():
        log_message("WARNING: Empty company_name detected in generate_queries!")
        # Try to extract company name from section description
        if hasattr(section, 'description') and section.description:
            # Look for a company name pattern - assuming it's at the beginning of the description
            match = re.search(r'([A-Z][a-zA-Z0-9_\' ]+)\'s', section.description)
            if match:
                extracted_company = match.group(1).strip()
                log_message(f"RECOVERY: Extracted company name '{extracted_company}' from section description")
                company_name = extracted_company
            else:
                log_message("ERROR: Could not extract company name from section description")
                company_name = "Danone"  # Hardcoded fallback as last resort
                log_message(f"FALLBACK: Using hardcoded company name '{company_name}'")
    
    number_of_queries = 8  # Increased from 5 to get more comprehensive data
    structured_llm = llm.with_structured_output(Queries)
    
    # Special handling for Product Category Analysis to focus on company's specific products
    if section.name == "Product Category Analysis":
        # Generate brand-specific queries if brands were discovered
        brand_specific_prompts = ""
        if company_brands:
            # Limit to top 5 brands for focused queries
            top_brands = company_brands[:6] if len(company_brands) > 5 else company_brands
            brand_specific_prompts = "Additional information about discovered brands:"
            for idx, brand in enumerate(top_brands):
                brand_name = brand.get("brand_name", "")
                category = brand.get("category", "")
                if brand_name and category:
                    brand_specific_prompts += f"\n{idx+1}. Brand: {brand_name} (Category: {category})"
            
            brand_specific_prompts += "\n\nInclude specific queries for these discovered brands."
        
        query_prompt = f"""
        You are generating search queries to research {company_name}'s PRODUCT CATEGORIES during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        {brand_specific_prompts}
        
        STEP 1: First, we need to identify {company_name}'s specific product categories and portfolio.
        
        Generate {number_of_queries} detailed and specific search queries to gather information about {company_name}'s SPECIFIC PRODUCT CATEGORIES and their performance during {time_period}.
        
        Include queries that specifically research:
        1. Complete list of all product categories and brands in {company_name}'s portfolio with revenue breakdown
        2. {company_name}'s fastest growing product categories with exact growth percentages and market size figures
        3. {company_name}'s market share data by product category with precise percentages versus named competitors (Chobani, General Mills, Pepsi, etc.)
        4. {company_name}'s underperforming product categories with specific decline rates and market share losses
        5. New product innovations and launches by {company_name} with sales projections and initial performance data
        6. {company_name}'s core product lines' performance metrics including volume, value, and market share evolution
        7. Category-specific profit margin data for {company_name}'s product portfolio compared to industry benchmarks
        8. Analyst reports with product category breakdowns for {company_name} containing specific revenue and growth metrics
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Reference specific product categories or brands owned by {company_name} when known
        3. Reference the time period "{time_period}" where relevant
        4. Include terms like "revenue figures," "market share percentages," "growth rates," "sales data," "profit margins"
        5. Target exact numerical data whenever possible
        6. Include competitor names for comparative queries
        7. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """
    elif section.name == "Brand Portfolio Analysis":
        # Generate brand-specific queries if brands were discovered
        brand_specific_prompts = ""
        if company_brands:
            brand_specific_prompts = "Additional information about discovered brands:"
            for idx, brand in enumerate(company_brands):
                brand_name = brand.get("brand_name", "")
                category = brand.get("category", "")
                if brand_name:
                    brand_specific_prompts += f"\n{idx+1}. Brand: {brand_name}" + (f" (Category: {category})" if category else "")
            
            brand_specific_prompts += "\n\nGenerate specific queries for each of these discovered brands."
        
        query_prompt = f"""
        You are generating search queries to research {company_name}'s BRAND PORTFOLIO during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        {brand_specific_prompts}
        
        Generate 15 detailed and specific search queries to gather information about {company_name}'s brand portfolio metrics during {time_period}.
        
        Include queries that specifically research:
        1. {company_name} brand portfolio overview complete list of all U.S. brands with revenue and growth figures {time_period}
        2. {company_name} brand-level revenues and growth exact percentages by brand {time_period}
        3. {company_name} target audience demographics and positioning strategy by brand {time_period}
        4. {company_name} marketing strategy by brand with sampling and experiential initiatives {time_period}
        5. {company_name} SWOT analysis of major brands in U.S. market {time_period}
        6. {company_name} top-performing U.S. brands with specific growth metrics {time_period}
        7. {company_name} underperforming U.S. brands with decline rates and challenges {time_period}
        8. {company_name} brand portfolio strategy and positioning matrix {time_period}
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Include specific brand names owned by {company_name} when possible
        3. Include terms like "brand portfolio," "brand-level revenue," "target audience," "positioning"
        4. Target exact numerical data such as "brand growth rates," "market share percentages," "demographic data"
        5. Include terms related to sampling, marketing strategy, and brand positioning
        6. Reference the time period "{time_period}" where relevant
        7. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """
    elif section.name == "Financial Performance":
        query_prompt = f"""
        You are generating search queries to research {company_name}'s FINANCIAL PERFORMANCE during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        Generate {number_of_queries} detailed and specific search queries to gather information about {company_name}'s financial metrics during {time_period}.
        
        Include queries that specifically research:
        1. {company_name} {time_period} revenue breakdown with exact figures by region, channel, and category in U.S. market
        2. {company_name} {time_period} quarterly earnings reports with detailed profitability and margin data
        3. {company_name} {time_period} U.S. market volume growth vs. pricing mix with specific percentages
        4. {company_name} {time_period} marketing expenditure and promotional spend exact figures in U.S. market
        5. {company_name} {time_period} gross margin, operating margin, net profit margin percentage changes
        6. {company_name} {time_period} earnings call transcript with specific commentary on U.S. market drivers and headwinds
        7. {company_name} {time_period} investor presentation with U.S. market financial metrics
        8. Analyst reports on {company_name} {time_period} financial performance in U.S. market with specific numbers
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Include terms like "exact revenue figures," "margin percentages," "marketing spend," "profitability metrics"
        3. Focus specifically on U.S. market performance whenever possible
        4. Target brand-level financial data when available (not just company-wide performance)
        5. Include terms seeking quantified drivers and headwinds (e.g., "volume drivers," "margin headwinds")
        6. Reference the time period "{time_period}" where relevant
        7. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """
    elif section.name == "Competitive Benchmarking & Market Share":
        query_prompt = f"""
        You are generating search queries to research {company_name}'s COMPETITIVE POSITIONING during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        Generate {number_of_queries} detailed and specific search queries to gather information about {company_name}'s competitive standing during {time_period}.
        
        Include queries that specifically research:
        1. {company_name} vs [specific competitor names] market share percentages {time_period}
        2. {company_name} competitive position industry rankings with exact numbers {time_period}
        3. {company_name} gaining market share from competitors specific percentage {time_period}
        4. Industry analyst reports comparing {company_name} to competitors with metrics {time_period}
        5. {company_name} product pricing comparison vs competitors with exact figures
        6. {company_name} promotional effectiveness vs competitors measurable results {time_period}
        7. {company_name} and competitors distribution channel performance numbers {time_period}
        8. {company_name} vs industry average growth rates specific percentages {time_period}
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Include at least one named competitor when appropriate
        3. Include terms like "versus," "compared to," "market share," "percentage," "outperformed"
        4. Reference the time period "{time_period}" where relevant
        5. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """
    elif section.name == "Marketing Strategy & Consumer Targeting":
        query_prompt = f"""
        You are generating search queries to research {company_name}'s MARKETING STRATEGY during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        Generate {number_of_queries} detailed and specific search queries to gather information about {company_name}'s marketing strategies and consumer targeting in the U.S. market.
        
        Include queries that specifically research:
        1. {company_name} {time_period} marketing spend trends with exact budget figures by channel
        2. {company_name} {time_period} target consumer segments with demographic and behavioral data
        3. {company_name} {time_period} sampling and experiential marketing initiatives metrics
        4. {company_name} {time_period} direct quotes from executives about consumer targeting
        5. {company_name} {time_period} digital vs. traditional marketing allocation percentages
        6. {company_name} {time_period} marketing ROI metrics and campaign effectiveness data
        7. {company_name} {time_period} consumer insights related to in-person product sampling
        8. {company_name} {time_period} marketing strategy for specific U.S. consumer demographics
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Include terms like "marketing budget," "consumer targeting," "sampling strategy," "experiential marketing"
        3. Target exact numerical data such as "spend percentages," "ROI metrics," "conversion rates"
        4. Include terms seeking direct executive quotes and statements about marketing strategy
        5. Focus on U.S. market-specific marketing activities when possible
        6. Reference the time period "{time_period}" where relevant
        7. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """
    elif section.name == "Retail Channel Performance":
        query_prompt = f"""
        You are generating search queries to research {company_name}'s RETAIL CHANNEL PERFORMANCE during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        Generate {number_of_queries} detailed and specific search queries to gather information about {company_name}'s performance across retail channels during {time_period}.
        
        Include queries that specifically research:
        1. {company_name} sales breakdown by retail channel exact percentages {time_period}
        2. {company_name} e-commerce vs brick-and-mortar sales figures {time_period}
        3. {company_name} performance in Walmart with specific numbers {time_period}
        4. {company_name} market share in convenience store channel exact percentage {time_period}
        5. {company_name} dollar store distribution metrics and performance {time_period}
        6. {company_name} vs competitors retail shelf space allocation percentages
        7. {company_name} club store sales growth exact figures {time_period}
        8. {company_name} retail media network investment figures and ROI {time_period}
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Name specific retailers (Walmart, Target, Kroger, Amazon, etc.) when appropriate
        3. Include terms like "sales," "percentage," "growth," "distribution," "metrics"
        4. Reference the time period "{time_period}" where relevant
        5. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """

    elif section.name == "U.S. Market & Regional Insights":
        query_prompt = f"""
        You are generating search queries to research {company_name}'s U.S. MARKET AND REGIONAL PERFORMANCE during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        Generate {number_of_queries} detailed and specific search queries to gather information about {company_name}'s performance across different U.S. regions and market segments.
        
        Include queries that specifically research:
        1. {company_name} {time_period} U.S. market retail vs. foodservice performance with exact figures and percentages
        2. {company_name} {time_period} regional sales breakdowns within U.S. (Northeast, Southeast, Midwest, West, etc.)
        3. {company_name} {time_period} struggling brands or product segments in U.S. market with specific decline rates
        4. {company_name} {time_period} growth in out-of-home consumption channels with specific metrics
        5. {company_name} {time_period} U.S. regional market challenges and response strategies with measurable results
        6. {company_name} {time_period} U.S. market share data by region and distribution channel
        7. {company_name} {time_period} U.S. consumer behavior patterns affecting regional performance
        8. Analyst reports on {company_name}'s {time_period} U.S. regional performance differences
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Specify "U.S. market" or U.S. regions (e.g., "Northeast," "Southern states," "California market")
        3. Include terms like "regional performance," "market breakdown," "geographic sales data," "channel-specific metrics"
        4. Target exact numerical data for regional differences and channel performance
        5. Include terms related to retail vs. foodservice, struggling segments, and out-of-home consumption
        6. Reference the time period "{time_period}" where relevant
        7. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """
    else:
        # Include company name and time period in the prompt for more specific queries
        query_prompt = f"""
        You are generating search queries to research {section.name} for {company_name} during {time_period}.
        
        SECTION TOPIC: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        Generate {number_of_queries} detailed and specific search queries to gather information about {company_name}'s {section.name.lower()} during {time_period}.
        
        Your queries should focus on finding:
        1. Precise numerical data and statistics about {company_name}'s performance
        2. Exact revenue figures, growth percentages, and financial metrics
        3. Market share numbers and competitive comparisons with specific percentages
        4. Analyst reports with detailed breakdowns of {company_name}'s performance
        5. Exact figures from earnings reports, investor presentations, and financial disclosures
        6. Named competitor comparisons with specific metrics
        7. Industry rankings with exact positioning
        8. Performance metrics broken down by product lines, regions, or channels
        
        Each query should:
        1. Include the company name "{company_name}" explicitly
        2. Reference the time period "{time_period}" where relevant
        3. Include terms like "percentage," "figures," "metrics," "exact," "numbers," "dollars," "growth rate"
        4. Be specific to one aspect of the section topic
        5. Be formatted for a web search engine
        
        FORMAT: Return a list of search queries in a structured format with a 'queries' field containing the list of query objects.
        """
    
    user_instruction = "Generate search queries on the provided topic."
    search_queries = structured_llm.invoke([
        SystemMessage(content=query_prompt),
        HumanMessage(content=user_instruction)
    ])
    
    # Log the actual queries for debugging
    log_message(f'LOGGING: Generated {len(search_queries.queries)} search queries:')
    for i, query in enumerate(search_queries.queries):
        log_message(f'LOGGING: Query {i+1}: {query.search_query}')
    
    log_message(f'--- Generating Search Queries for Section: {section.name} Completed ---')
    return {"search_queries": search_queries.queries}


async def search_web(state: SectionState):
    """Search the web for each query, then return a list of raw sources and a formatted string of sources."""
    search_queries = state["search_queries"]
    section = state["section"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    
    log_message('--- Searching Web for Queries ---')
    log_message(f'LOGGING: Running {len(search_queries)} web searches for {company_name}, section {section.name}')
    query_list = [query.search_query for query in search_queries]
    search_docs = await run_search_queries(query_list, num_results=6, include_raw_content=True)

    # Log search results for debugging
    log_message(f'LOGGING: Received {len(search_docs)} search results')
    for i, doc in enumerate(search_docs[:3]):  # Log first 3 results to avoid excessive output
        log_message(f'LOGGING: Result {i+1} Title: {doc.get("title", "No title")}')
        log_message(f'LOGGING: Result {i+1} URL: {doc.get("link", "No link")}')
        log_message(f'LOGGING: Result {i+1} Snippet: {doc.get("snippet", "No snippet")[:100]}...')
        log_message(f'LOGGING: Company name "{company_name}" appears {str(doc.get("content", "")).lower().count(company_name.lower())} times in content')
    
    section.search_docs = search_docs  

    search_context = format_search_query_results(search_docs, max_tokens=4000, include_raw_content=True)
    log_message('--- Searching Web for Queries Completed ---')
    return {
        "source_str": search_context,
        "search_docs": search_docs,
        "section": section,
        "company_name": company_name,
        "time_period": time_period
    }


async def collect_evidence(state: SectionState):
    """Collect specific evidence points from search results."""
    search_queries = state["search_queries"]
    section = state["section"]
    company_brands = state["company_brands"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    
    log_message(f'--- Collecting Evidence for Section: {section.name} ---')
    
    search_docs = await run_search_queries(search_queries, num_results=3, include_raw_content=True)
    # Also store in the section, merging with any existing docs
    section.search_docs.extend(search_docs)

    search_context = format_search_query_results(search_docs, max_tokens=6000, include_raw_content=True)
    
    # Log the search context to see what content we're working with
    log_message(f'LOGGING: Search context contains {search_context.lower().count(company_name.lower())} mentions of {company_name}')
    if search_context.lower().count(company_name.lower()) < 5:
        log_message(f'WARNING: Very few mentions of {company_name} in search results. This may lead to a generic report.')
    
    # Update to refer to predefined subsections
    subsection_names = [subsec.title for subsec in section.subsections]
    subsection_list = ", ".join(subsection_names)
    
    # Add company relevance check to all evidence collection prompts
    company_relevance_check = f"""
    EXTREMELY IMPORTANT: Each evidence point MUST be related to {company_name} specifically. 
    
    CRITICAL INSTRUCTIONS:
    1. EVERY evidence point MUST explain the "why" behind the numbers by connecting cause and effect relationships (e.g., "Revenue increased by 12% due to new product launches and expanded distribution channels")
    2. ONLY extract information if it can be related to {company_name}
    3. NEVER include general industry information unless it directly compares to {company_name}
    4. NEVER include information about competitors except when directly compared to {company_name}
    5. EVERY evidence point MUST include AT LEAST TWO specific numerical metrics
    6. EVERY evidence point MUST include a clear, verifiable source
    
    
    If the search results contain little relevant information about {company_name}, focus only on those few pieces that are relevant 
    rather than including generic industry information or information about competitor companies.
    """
    
    # Different prompts for different section types
    if section.name == "Product Category Analysis":
        evidence_prompt = f"""
        You are a FINANCIAL ANALYST extracting precise data about {company_name}'s PRODUCT CATEGORIES during {time_period}.
        
        SECTION: {section.name}
        DESCRIPTION: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        PRODUCT CATEGORY: {company_brands}
        
        {company_relevance_check}
        
        From the search results below, extract 15-20 specific evidence points about {company_name}'s product portfolio and categories.
        
        PRIORITIZE THE FOLLOWING DATA TYPES:
        1. EXACT REVENUE FIGURES - Extract specific dollar amounts for each product category
        2. PRECISE MARKET SHARE DATA - Extract exact percentage of market share by category with competitor comparisons
        3. GROWTH/DECLINE RATES - Extract specific YoY or QoQ growth/decline percentages for each category
        4. VOLUME VS VALUE METRICS - Extract data showing unit volume changes vs. revenue changes
        5. PRICING STRATEGY IMPACT - Extract data on pricing changes and their effect on category performance
        6. PROFITABILITY BY CATEGORY - Extract gross margin or operating margin data by product category
        7. INNOVATION CONTRIBUTION - Extract revenue/share attributed to new product launches
        8. COMPETITOR BENCHMARKING - Extract direct comparisons between {company_name}'s categories and specific competitors
        
        Each evidence point MUST:
        1. EVERY evidence point MUST explain the "why" behind the numbers by connecting cause and effect relationships (e.g., "Revenue increased by 12% due to new product launches and expanded distribution channels")
        2. Focus on a SPECIFIC PRODUCT CATEGORY (e.g., yogurt, plant-based products, water, etc.)
        3. Include AT LEAST TWO SPECIFIC NUMBERS, PERCENTAGES, OR FINANCIAL METRICS
        4. Include the precise source (publication name, website, analyst report, earnings call)
        5. Include a DATE or TIME PERIOD for the data point
        6. Be assigned to one of these subsections: {subsection_list}
        
        FOR EACH SUBSECTION SPECIFICALLY:
        
        For "High-Growth Categories" subsection:
        - Extract EXACT GROWTH PERCENTAGES (e.g., "grew by 15.2% YoY") for {company_name}'s fastest growing categories
        - Include DOLLAR FIGURES showing the size of these categories (e.g., "$1.2 billion category")
        - Identify GROWTH DRIVERS with quantified impact (e.g., "price increases accounted for 60% of growth")
        - Include TREND DATA showing acceleration or deceleration in growth
        - Compare growth to CATEGORY AVERAGE (e.g., "outpaced category growth by 300 basis points")
        
        For "Category-Level Market Share" subsection:
        - Extract PRECISE MARKET SHARE PERCENTAGES for each major category (e.g., "holds 28.4% market share")
        - Include SHARE CHANGE data (e.g., "gained 1.2 percentage points YoY")
        - Compare directly to NAMED COMPETITORS with their share data (e.g., "vs. Competitor X's 18.2%")
        - Include REGIONAL VARIATIONS in market share when available
        - Note CHANNEL-SPECIFIC share positions (e.g., "42% share in convenience channel")
        
        For "Underperforming Categories" subsection:
        - Extract EXACT DECLINE RATES for struggling categories (e.g., "declined 7.8% YoY")
        - Include DOLLAR VALUE of the decline (e.g., "representing a $45M loss")
        - Identify SPECIFIC FACTORS causing underperformance with quantified impact
        - Include COMPETITOR GAINS at {company_name}'s expense with specific figures
        - Note REMEDIATION STRATEGIES with targeted metrics (e.g., "aims to return to growth by Q3")
        
        For "Emerging Trends" subsection:
        - Extract SPECIFIC SALES FORECASTS for new or emerging categories
        - Include NEW PRODUCT PERFORMANCE metrics from recent launches
        - Quantify CONSUMER ADOPTION RATES for new product types or formats
        - Include INNOVATION PIPELINE details with projected revenue contribution
        - Note CATEGORY DISRUPTION potential with quantified impact projections
        
        DO NOT include generic industry information unless it directly relates to {company_name}'s products.
        DO NOT create or summarize data points - only extract FACTUAL, VERIFIABLE metrics that appear in the search results.
        
        SEARCH RESULTS:
        {search_context}
        
        FORMAT: Return a list of evidence points in a structured format with an 'evidence_points' field containing the list.
        """
    elif section.name == "Financial Performance":
        evidence_prompt = f"""
        You are a FINANCIAL ANALYST extracting precise financial data about {company_name}'s U.S. market performance during {time_period}.
        
        SECTION: {section.name}
        DESCRIPTION: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        {company_relevance_check}
        
        From the search results below, extract 15-20 specific evidence points about {company_name}'s financial performance in the U.S. market.
        
        PRIORITIZE THE FOLLOWING DATA TYPES:
        1. REVENUE METRICS - Extract exact dollar amounts, growth percentages, and breakdowns by category/brand
        2. VOLUME VS PRICING - Extract specific volume growth/decline percentages separated from pricing impacts
        3. MARGIN DATA - Extract precise margin percentages (gross, operating, net) with basis point changes
        4. MARKETING EXPENDITURE - Extract specific marketing and promotional spend figures with ROI metrics
        5. FINANCIAL DRIVERS - Extract quantified positive factors driving financial performance
        6. FINANCIAL HEADWINDS - Extract quantified challenges impacting financial results
        7. EXECUTIVE COMMENTARY - Extract direct quotes from executives explaining financial results
        8. BRAND-LEVEL FINANCIALS - Extract specific financial data for individual brands in the U.S. market
        
        Each evidence point MUST:
        1. Include AT LEAST TWO SPECIFIC NUMBERS, PERCENTAGES, OR FINANCIAL METRICS
        2. Include quarterly designations (Q1, Q2, etc.) or specify fiscal/calendar year
        3. Include the precise source (earnings report, analyst call, investor presentation)
        4. Specify whether the data is for the total U.S. market or specific regions/channels
        5. Be assigned to one of these subsections: {subsection_list}
        
        FOR EACH SUBSECTION SPECIFICALLY:
        
        For "Revenue Growth & Volume/Mix" subsection:
        - Extract EXACT REVENUE FIGURES in U.S. dollars (e.g., "$3.4 billion U.S. revenue")
        - Separate VOLUME GROWTH from PRICING EFFECTS with percentages (e.g., "volume up 2.1%, pricing up 4.3%")
        - Include CATEGORY-SPECIFIC revenue growth (e.g., "yogurt category grew 5.6% in U.S.")
        - Extract MARKET SHARE CHANGES tied to revenue (e.g., "gained 0.8 percentage points in market share")
        - Include SALES CHANNEL BREAKDOWNS (e.g., "retail channel grew 7.2%, foodservice declined 3.1%")
        
        For "Profitability & Margins" subsection:
        - Extract PRECISE MARGIN PERCENTAGES with basis point changes (e.g., "gross margin expanded 85 basis points to 42.3%")
        - Include COST FACTORS affecting margins with quantified impacts (e.g., "input costs increased 12%, reducing margins by 40 bps")
        - Extract EFFICIENCY METRICS and savings figures (e.g., "productivity initiatives delivered $45M in savings")
        - Include OPERATING INCOME figures with YoY changes (e.g., "operating income grew 7.8% to $780M")
        - Extract SEGMENT-LEVEL profitability data when available (e.g., "U.S. dairy segment operating margin of 18.2%")
        
        For "Marketing & Promotional Spend" subsection:
        - Extract EXACT MARKETING SPEND in dollars and as percentage of sales (e.g., "$450M in marketing, 12.5% of sales")
        - Include PROMOTIONAL EFFECTIVENESS metrics (e.g., "promotions delivered 2.3x ROI, up from 1.8x")
        - Extract MEDIA MIX ALLOCATION percentages (e.g., "digital channels now represent 65% of marketing budget")
        - Include BRAND-SPECIFIC marketing investments (e.g., "increased Brand X marketing by 22% to $85M")
        - Extract PRICING & PROMOTION trade-offs (e.g., "reduced promotional depth by 15%, maintaining 3.2% volume growth")
        
        For "Key Drivers & Headwinds" subsection:
        - Extract QUANTIFIED GROWTH DRIVERS (e.g., "premium portfolio grew 14.2%, adding $120M in incremental revenue")
        - Include SPECIFIC CHALLENGES with measurable impacts (e.g., "supply chain disruptions reduced revenue by $35M")
        - Extract COMPETITOR ACTIONS affecting performance (e.g., "lost 1.2 share points to Competitor Y's pricing actions")
        - Include MACROECONOMIC FACTORS with quantified impacts (e.g., "inflation reduced consumer purchasing power by 7.5%")
        - Extract FORWARD-LOOKING STATEMENTS with numerical projections (e.g., "expects costs to moderate by 150 bps in next quarter")
        
        DO NOT include vague financial statements without specific numbers.
        DO NOT create or summarize data points - only extract FACTUAL, VERIFIABLE financial metrics that appear in the search results.
        
        SEARCH RESULTS:
        {search_context}
        
        FORMAT: Return a list of evidence points in a structured format with an 'evidence_points' field containing the list.
        """
    elif section.name == "Competitive Benchmarking & Market Share":
        evidence_prompt = f"""
        You are a COMPETITIVE INTELLIGENCE ANALYST extracting precise comparative data about {company_name} and its competitors during {time_period}.
        
        SECTION: {section.name}
        DESCRIPTION: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        From the search results below, extract 15-20 specific evidence points comparing {company_name} to its direct competitors.
        
        PRIORITIZE THE FOLLOWING:
        1. DIRECT MARKET SHARE COMPARISONS - Extract exact market share percentages between {company_name} and named competitors
        2. PERFORMANCE DIFFERENTIALS - Extract specific metrics showing how {company_name} outperforms or underperforms competitors
        3. COMPETITIVE POSITIONING - Extract precise data about pricing, distribution, or quality advantages/disadvantages
        4. CONSUMER PREFERENCE METRICS - Extract specific data from consumer surveys showing brand preference percentages
        5. GROWTH RATE DIFFERENCES - Extract exact YoY growth rate comparisons between {company_name} and competitors
        
        Each evidence point MUST:
        1. EXPLICITLY NAME a specific competitor being compared to {company_name}
        2. Include AT LEAST ONE SPECIFIC NUMBER, PERCENTAGE, or METRIC for both {company_name} AND the competitor
        3. Include the precise source (market research report, industry analysis, financial publication)
        4. Include EXACT DATES for the comparative data
        5. Indicate which of these subsections it belongs to: {subsection_list}
        
        For "Market Share Analysis" - focus on EXACT PERCENTAGE COMPARISONS of market share between {company_name} and named competitors
        For "Competitive Advantages" - focus on SPECIFIC METRICS where {company_name} OUTPERFORMS competitors
        For "Competitive Threats" - focus on PRECISE AREAS where competitors are GAINING GROUND with exact figures
        For "Industry Positioning" - focus on EXACT RANKINGS and QUANTIFIABLE DIFFERENTIATORS
        
        DO NOT include vague comparisons without specific numbers for both {company_name} AND competitors.
        DO NOT create or summarize data points - only extract FACTUAL, VERIFIABLE comparative metrics that appear in the search results.
        
        SEARCH RESULTS:
        {search_context}
        
        FORMAT: Return a list of evidence points in a structured format with an 'evidence_points' field containing the list.
        """
    elif section.name == "Brand Portfolio Analysis":
        evidence_prompt = f"""
        You are a BRAND STRATEGIST extracting precise data about {company_name}'s BRAND PORTFOLIO in the U.S. market during {time_period}.
        
        SECTION: {section.name}
        DESCRIPTION: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        COMPANY BRANDS: {company_brands}
        
        {company_relevance_check}
        
        From the search results below, extract 15-20 specific evidence points about {company_name}'s major U.S. brands and their performance.
        
        PRIORITIZE THE FOLLOWING DATA TYPES:
        1. BRAND REVENUE FIGURES - Extract exact dollar amounts and growth rates for specific brands
        2. TARGET AUDIENCE DATA - Extract detailed demographic and psychographic profiles for each brand
        3. BRAND POSITIONING INFORMATION - Extract specific positioning statements and brand attributes
        4. MARKETING STRATEGY - Extract precise data on marketing approaches for each brand, especially sampling
        5. SWOT ANALYSIS ELEMENTS - Extract specific strengths, weaknesses, opportunities, and threats for brands
        6. COMPETITIVE POSITIONING - Extract market share data comparing brands to specific competitors
        7. BRAND PERFORMANCE METRICS - Extract KPIs like household penetration, purchase frequency, loyalty rates
        8. SAMPLING INITIATIVES - Extract specific data on sampling and experiential marketing by brand
        
        Each evidence point MUST:
        1. Focus on a SPECIFIC NAMED BRAND owned by {company_name}
        2. Include AT LEAST TWO SPECIFIC NUMBERS, PERCENTAGES, OR BRAND METRICS
        3. Include the precise source (brand reports, marketing documents, executive statements)
        4. Include a DATE or TIME PERIOD for the brand data
        5. Be assigned to one of these subsections: {subsection_list}
        
        FOR EACH SUBSECTION SPECIFICALLY:
        
        For "Brand Portfolio Overview" subsection:
        - Extract COMPLETE BRAND PORTFOLIO listings with category classifications
        - Include BRAND HIERARCHY information (premium, mainstream, value tiers)
        - Extract BRAND ARCHITECTURE details (endorsed brands, standalone brands, sub-brands)
        - Include BRAND RELATIONSHIPS and positioning relative to each other
        - Extract PORTFOLIO STRATEGY insights with executive rationale
        
        For "Brand-Level Revenues & Growth" subsection:
        - Extract EXACT REVENUE FIGURES by brand (e.g., "Brand X generated $340M in U.S. sales")
        - Include GROWTH PERCENTAGES with YoY comparisons (e.g., "Brand Y grew 12.8% YoY")
        - Extract VOLUME VS VALUE growth components (e.g., "8.2% price growth, 4.6% volume growth")
        - Include MARKET SHARE data by brand (e.g., "Brand Z holds 15.3% share in its category")
        - Extract SALES CHANNEL BREAKDOWN by brand (e.g., "60% of Brand X revenue from grocery, 25% from mass, 15% from club")
        
        For "Target Audience & Positioning" subsection:
        - Extract SPECIFIC DEMOGRAPHIC PROFILES for each brand (e.g., "Brand A targets women 25-45 with household income >$75K")
        - Include PSYCHOGRAPHIC DETAILS with percentages (e.g., "65% of Brand B consumers prioritize health and wellness")
        - Extract POSITIONING STATEMENTS verbatim when available
        - Include BRAND ATTRIBUTE ratings and consumer perception data
        - Extract COMPETITIVE DIFFERENTIATION points with supporting metrics
        
        For "Marketing Strategy & Sampling Focus" subsection:
        - Extract BRAND-SPECIFIC MARKETING APPROACHES with budget allocations
        - Include SAMPLING PROGRAM details with specific metrics (e.g., "distributed 2.5M samples, achieving 42% conversion")
        - Extract EXPERIENTIAL MARKETING initiatives with ROI data
        - Include CHANNEL-SPECIFIC marketing tactics by brand
        - Extract CONSUMER ENGAGEMENT metrics from sampling/experiential activities
        
        For "SWOT Analysis (U.S. Market)" subsection:
        - Extract QUANTIFIED STRENGTHS with supporting metrics (e.g., "Brand C's 85% name recognition, highest in category")
        - Include SPECIFIC WEAKNESSES with measurable impact (e.g., "Brand D's limited distribution in convenience, reaching only 45% of potential outlets")
        - Extract SPECIFIC OPPORTUNITIES with market sizing (e.g., "potential to grow household penetration by 3.5 percentage points, worth $60M")
        - Include QUANTIFIED THREATS with competitive context (e.g., "Competitor X gained 2.2 share points against Brand E")
        - Extract RISK ASSESSMENT metrics with probability and impact ratings
        
        DO NOT include vague brand statements without specific metrics.
        DO NOT create or summarize data points - only extract FACTUAL, VERIFIABLE brand metrics that appear in the search results.
        
        SEARCH RESULTS:
        {search_context}
        
        FORMAT: Return a list of evidence points in a structured format with an 'evidence_points' field containing the list.
        """
    elif section.name == "Brand Performance":
        evidence_prompt = f"""
        You are a BRAND STRATEGIST extracting precise data about {company_name}'s BRAND PORTFOLIO in the U.S. market during {time_period}.
        
        SECTION: Brand Portfolio Analysis
        DESCRIPTION: Provide details on all major brands owned by {company_name} in the U.S. market: revenue/growth, target audience/positioning, marketing strategy (especially regarding sampling), and a SWOT analysis if possible.
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        {company_relevance_check}
        
        From the search results below, extract 15-20 specific evidence points about {company_name}'s major U.S. brands and their performance.
        
        PRIORITIZE THE FOLLOWING DATA TYPES:
        1. BRAND REVENUE FIGURES - Extract exact dollar amounts and growth rates for specific brands
        2. TARGET AUDIENCE DATA - Extract detailed demographic and psychographic profiles for each brand
        3. BRAND POSITIONING INFORMATION - Extract specific positioning statements and brand attributes
        4. MARKETING STRATEGY - Extract precise data on marketing approaches for each brand, especially sampling
        5. SWOT ANALYSIS ELEMENTS - Extract specific strengths, weaknesses, opportunities, and threats for brands
        6. COMPETITIVE POSITIONING - Extract market share data comparing brands to specific competitors
        7. BRAND PERFORMANCE METRICS - Extract KPIs like household penetration, purchase frequency, loyalty rates
        8. SAMPLING INITIATIVES - Extract specific data on sampling and experiential marketing by brand
        
        Each evidence point MUST:
        1. Focus on a SPECIFIC NAMED BRAND owned by {company_name}
        2. Include AT LEAST TWO SPECIFIC NUMBERS, PERCENTAGES, OR BRAND METRICS
        3. Include the precise source (brand reports, marketing documents, executive statements)
        4. Include a DATE or TIME PERIOD for the brand data
        5. Be assigned to one of these subsections: Brand Portfolio Overview, Brand-Level Revenues & Growth, Target Audience & Positioning, Marketing Strategy & Sampling Focus, SWOT Analysis (U.S. Market)
        
        FOR EACH SUBSECTION SPECIFICALLY:
        
        For "Brand Portfolio Overview" subsection:
        - Extract COMPLETE BRAND PORTFOLIO listings with category classifications
        - Include BRAND HIERARCHY information (premium, mainstream, value tiers)
        - Extract BRAND ARCHITECTURE details (endorsed brands, standalone brands, sub-brands)
        - Include BRAND RELATIONSHIPS and positioning relative to each other
        - Extract PORTFOLIO STRATEGY insights with executive rationale
        
        For "Brand-Level Revenues & Growth" subsection:
        - Extract EXACT REVENUE FIGURES by brand (e.g., "Brand X generated $340M in U.S. sales")
        - Include GROWTH PERCENTAGES with YoY comparisons (e.g., "Brand Y grew 12.8% YoY")
        - Extract VOLUME VS VALUE growth components (e.g., "8.2% price growth, 4.6% volume growth")
        - Include MARKET SHARE data by brand (e.g., "Brand Z holds 15.3% share in its category")
        - Extract SALES CHANNEL BREAKDOWN by brand (e.g., "60% of Brand X revenue from grocery, 25% from mass, 15% from club")
        
        For "Target Audience & Positioning" subsection:
        - Extract SPECIFIC DEMOGRAPHIC PROFILES for each brand (e.g., "Brand A targets women 25-45 with household income >$75K")
        - Include PSYCHOGRAPHIC DETAILS with percentages (e.g., "65% of Brand B consumers prioritize health and wellness")
        - Extract POSITIONING STATEMENTS verbatim when available
        - Include BRAND ATTRIBUTE ratings and consumer perception data
        - Extract COMPETITIVE DIFFERENTIATION points with supporting metrics
        
        For "Marketing Strategy & Sampling Focus" subsection:
        - Extract BRAND-SPECIFIC MARKETING APPROACHES with budget allocations
        - Include SAMPLING PROGRAM details with specific metrics (e.g., "distributed 2.5M samples, achieving 42% conversion")
        - Extract EXPERIENTIAL MARKETING initiatives with ROI data
        - Include CHANNEL-SPECIFIC marketing tactics by brand
        - Extract CONSUMER ENGAGEMENT metrics from sampling/experiential activities
        
        For "SWOT Analysis (U.S. Market)" subsection:
        - Extract QUANTIFIED STRENGTHS with supporting metrics (e.g., "Brand C's 85% name recognition, highest in category")
        - Include SPECIFIC WEAKNESSES with measurable impact (e.g., "Brand D's limited distribution in convenience, reaching only 45% of potential outlets")
        - Extract SPECIFIC OPPORTUNITIES with market sizing (e.g., "potential to grow household penetration by 3.5 percentage points, worth $60M")
        - Include QUANTIFIED THREATS with competitive context (e.g., "Competitor X gained 2.2 share points against Brand E")
        - Extract RISK ASSESSMENT metrics with probability and impact ratings
        
        DO NOT include vague brand statements without specific metrics.
        DO NOT create or summarize data points - only extract FACTUAL, VERIFIABLE brand metrics that appear in the search results.
        
        SEARCH RESULTS:
        {search_context}
        
        FORMAT: Return a list of evidence points in a structured format with an 'evidence_points' field containing the list.
        """
    elif section.name == "Specific Brand Insights":
        evidence_prompt = f"""
        You are a BRAND STRATEGIST extracting precise data about {company_name}'s BRAND PORTFOLIO in the U.S. market during {time_period}.
        
        SECTION: Brand Portfolio Analysis
        DESCRIPTION: Provide details on all major brands owned by {company_name} in the U.S. market: revenue/growth, target audience/positioning, marketing strategy (especially regarding sampling), and a SWOT analysis if possible.
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        {company_relevance_check}
        
        From the search results below, extract 15-20 specific evidence points about {company_name}'s major U.S. brands and their performance.
        
        PRIORITIZE THE FOLLOWING DATA TYPES:
        1. BRAND REVENUE FIGURES - Extract exact dollar amounts and growth rates for specific brands
        2. TARGET AUDIENCE DATA - Extract detailed demographic and psychographic profiles for each brand
        3. BRAND POSITIONING INFORMATION - Extract specific positioning statements and brand attributes
        4. MARKETING STRATEGY - Extract precise data on marketing approaches for each brand, especially sampling
        5. SWOT ANALYSIS ELEMENTS - Extract specific strengths, weaknesses, opportunities, and threats for brands
        6. COMPETITIVE POSITIONING - Extract market share data comparing brands to specific competitors
        7. BRAND PERFORMANCE METRICS - Extract KPIs like household penetration, purchase frequency, loyalty rates
        8. SAMPLING INITIATIVES - Extract specific data on sampling and experiential marketing by brand
        
        Each evidence point MUST:
        1. Focus on a SPECIFIC NAMED BRAND owned by {company_name}
        2. Include AT LEAST TWO SPECIFIC NUMBERS, PERCENTAGES, OR BRAND METRICS
        3. Include the precise source (brand reports, marketing documents, executive statements)
        4. Include a DATE or TIME PERIOD for the brand data
        5. Be assigned to one of these subsections: Brand Portfolio Overview, Brand-Level Revenues & Growth, Target Audience & Positioning, Marketing Strategy & Sampling Focus, SWOT Analysis (U.S. Market)
        
        FOR EACH SUBSECTION SPECIFICALLY:
        
        For "Brand Portfolio Overview" subsection:
        - Extract COMPLETE BRAND PORTFOLIO listings with category classifications
        - Include BRAND HIERARCHY information (premium, mainstream, value tiers)
        - Extract BRAND ARCHITECTURE details (endorsed brands, standalone brands, sub-brands)
        - Include BRAND RELATIONSHIPS and positioning relative to each other
        - Extract PORTFOLIO STRATEGY insights with executive rationale
        
        For "Brand-Level Revenues & Growth" subsection:
        - Extract EXACT REVENUE FIGURES by brand (e.g., "Brand X generated $340M in U.S. sales")
        - Include GROWTH PERCENTAGES with YoY comparisons (e.g., "Brand Y grew 12.8% YoY")
        - Extract VOLUME VS VALUE growth components (e.g., "8.2% price growth, 4.6% volume growth")
        - Include MARKET SHARE data by brand (e.g., "Brand Z holds 15.3% share in its category")
        - Extract SALES CHANNEL BREAKDOWN by brand (e.g., "60% of Brand X revenue from grocery, 25% from mass, 15% from club")
        
        For "Target Audience & Positioning" subsection:
        - Extract SPECIFIC DEMOGRAPHIC PROFILES for each brand (e.g., "Brand A targets women 25-45 with household income >$75K")
        - Include PSYCHOGRAPHIC DETAILS with percentages (e.g., "65% of Brand B consumers prioritize health and wellness")
        - Extract POSITIONING STATEMENTS verbatim when available
        - Include BRAND ATTRIBUTE ratings and consumer perception data
        - Extract COMPETITIVE DIFFERENTIATION points with supporting metrics
        
        For "Marketing Strategy & Sampling Focus" subsection:
        - Extract BRAND-SPECIFIC MARKETING APPROACHES with budget allocations
        - Include SAMPLING PROGRAM details with specific metrics (e.g., "distributed 2.5M samples, achieving 42% conversion")
        - Extract EXPERIENTIAL MARKETING initiatives with ROI data
        - Include CHANNEL-SPECIFIC marketing tactics by brand
        - Extract CONSUMER ENGAGEMENT metrics from sampling/experiential activities
        
        For "SWOT Analysis (U.S. Market)" subsection:
        - Extract QUANTIFIED STRENGTHS with supporting metrics (e.g., "Brand C's 85% name recognition, highest in category")
        - Include SPECIFIC WEAKNESSES with measurable impact (e.g., "Brand D's limited distribution in convenience, reaching only 45% of potential outlets")
        - Extract SPECIFIC OPPORTUNITIES with market sizing (e.g., "potential to grow household penetration by 3.5 percentage points, worth $60M")
        - Include QUANTIFIED THREATS with competitive context (e.g., "Competitor X gained 2.2 share points against Brand E")
        - Extract RISK ASSESSMENT metrics with probability and impact ratings
        
        DO NOT include vague brand statements without specific metrics.
        DO NOT create or summarize data points - only extract FACTUAL, VERIFIABLE brand metrics that appear in the search results.
        
        SEARCH RESULTS:
        {search_context}
        
        FORMAT: Return a list of evidence points in a structured format with an 'evidence_points' field containing the list.
        """
    else:
        # Generic evidence prompt with emphasis on extracting numerical data
        evidence_prompt = f"""
        You are an INDUSTRY ANALYST extracting precise data about {company_name}'s {section.name.lower()} during {time_period}.
        
        SECTION: {section.name}
        DESCRIPTION: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        From the search results below, extract 15-20 specific evidence points about {company_name}'s {section.name.lower()}.
        
        PRIORITIZE THE FOLLOWING:
        1. EXACT NUMERICAL DATA - Extract specific numbers, percentages, dollar amounts, and growth rates
        2. COMPETITIVE COMPARISONS - Extract data that compares {company_name} to specific named competitors
        3. PERFORMANCE METRICS - Extract precise performance indicators with exact figures
        4. TIME-BASED TRENDS - Extract data showing changes over time with specific percentages
        5. FINANCIAL IMPACTS - Extract exact dollar amounts showing financial impact of strategies or challenges
        
        Each evidence point MUST:
        1. Focus SPECIFICALLY on {company_name}'s performance, not generic industry information
        2. Include AT LEAST ONE SPECIFIC NUMBER, PERCENTAGE, OR METRIC
        3. Include the precise source (e.g., publication name, website, analyst report, earnings call)
        4. Include a DATE or TIME PERIOD for the data point whenever available
        5. Indicate which of these subsections it belongs to: {subsection_list}
        
        DO NOT include vague statements without specific metrics.
        DO NOT create or summarize data points - only extract FACTUAL, VERIFIABLE metrics that appear in the search results.
        
        SEARCH RESULTS:
        {search_context}
        
        FORMAT: Return a list of evidence points in a structured format with an 'evidence_points' field containing the list.
        """
    llm = get_llm("o3-mini")
    structured_llm = llm.with_structured_output(EvidencePoints)
    evidence_points_container = structured_llm.invoke([
        SystemMessage(content=evidence_prompt),
        HumanMessage(content=f"Extract evidence points for {company_name}'s {section.name} during {time_period}.")
    ])
    
    evidence_points = evidence_points_container.evidence_points
    
    try:
        section.evidence_points = evidence_points
        log_message(f'--- Collected {len(evidence_points)} Evidence Points for Section: {section.name} ---')
    except Exception as e:
        log_message(f'Warning: Unable to store evidence points directly in section: {e}')
        log_message(f'--- Collected {len(evidence_points)} Evidence Points for Section: {section.name} ---')
    
    return {
        "evidence_points": evidence_points,
        "source_str": search_context,
        "section": section,
        "company_name": company_name,
        "time_period": time_period
    }


def map_evidence_to_subsections(state: SectionState):
    """
    Map collected evidence points to the predefined subsections.
    This replaces the organize_subsections function in the bottom-up approach.
    """
    section = state["section"]
    evidence_points = state["evidence_points"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    
    log_message(f'--- Mapping Evidence to Predefined Subsections for Section: {section.name} ---')
    
    # Sanity check - ensure we have evidence points
    if not evidence_points or len(evidence_points) == 0:
        log_message(f"Warning: No evidence points found for section {section.name}. Creating placeholder paragraphs.")
        # Create placeholder paragraphs for all subsections
        for subsection in section.subsections:
            subsection.paragraphs = [
                Paragraph(
                    main_idea=f"Overview of {subsection.title}",
                    points=[SubPoint(content=f"No specific evidence found for {company_name}'s {subsection.title} during {time_period}.", sources=["N/A"])],
                    synthesized_content=""
                )
            ]
        return {"section": section}
    
    # For each subsection, find relevant evidence points and create paragraphs
    for subsection in section.subsections:
        # Extract evidence relevant to this subsection
        subsection_title = subsection.title.lower()
        
        # Special handling for Product Category Analysis section
        if section.name == "Product Category Analysis":
            product_focus_instructions = f"""
            IMPORTANT: Focus specifically on {company_name}'s product categories and portfolio.
            
            For "High-Growth Categories" - focus ONLY on {company_name}'s fastest growing product categories and brands
            For "Category-Level Market Share" - focus ONLY on {company_name}'s share within specific product categories
            For "Underperforming Categories" - focus ONLY on {company_name}'s struggling product lines and categories
            For "Emerging Trends" - focus ONLY on new innovations within {company_name}'s product portfolio
            
            DO NOT include generic industry information unless it directly relates to {company_name}'s specific products.
            Each main idea MUST be about a specific {company_name} product category or brand.
            """
        else:
            product_focus_instructions = ""
        
        # Prompt to map evidence points to this specific subsection
        mapping_prompt = f"""
        You are analyzing evidence points for a market performance report on {company_name} during {time_period}.
        
        SECTION: {section.name}
        SUBSECTION: {subsection.title}
        DESCRIPTION: {section.description}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        {product_focus_instructions}
        
        Your task is to identify evidence points most relevant to this specific subsection.
        Then, organize these evidence points into logical main ideas (3-5 main ideas).
        
        Here are the available evidence points:
        {[f"- {e.fact} (Source: {e.source}, Relevance: {e.relevance})" for e in evidence_points]}
        
        For each main idea you identify:
        1. Give it a clear, concise title that relates to {company_name}'s performance
        2. List the evidence points that support this main idea
        
        FORMAT YOUR RESPONSE AS A JSON OBJECT with this structure:
        {{
            "main_ideas": [
                {{
                    "title": "Main Idea 1",
                    "evidence_indices": [0, 3, 5]  # Indices of the evidence points in the list above that support this idea
                }},
                {{
                    "title": "Main Idea 2",
                    "evidence_indices": [1, 4]
                }}
            ]
        }}
        """
        
        try:
            # Use LLM to map evidence points to main ideas for this subsection
            mapping_response = llm.invoke([
                SystemMessage(content=mapping_prompt),
                HumanMessage(content=f"Map evidence points to main ideas for the '{subsection.title}' subsection of {company_name}'s {time_period} report.")
            ])
            
            # Parse the response to extract the main ideas and evidence mappings
            response_text = mapping_response.content
            # Extract JSON part if wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
                
            mapping_data = json.loads(response_text)
            main_ideas = mapping_data.get("main_ideas", [])
            
            # Create paragraphs based on the main ideas
            paragraphs = []
            for idea in main_ideas:
                title = idea.get("title", "")
                evidence_indices = idea.get("evidence_indices", [])
                
                # Create SubPoint objects for the supporting evidence
                sub_points = []
                for idx in evidence_indices:
                    if 0 <= idx < len(evidence_points):
                        evidence = evidence_points[idx]
                        sub_points.append(
                            SubPoint(
                                content=evidence.fact,
                                sources=[evidence.source]
                            )
                        )
                
                # Create a Paragraph object for this main idea
                if sub_points:
                    paragraphs.append(
                        Paragraph(
                            main_idea=title,
                            points=sub_points,
                            synthesized_content=""
                        )
                    )
            
            # Update the subsection with the paragraphs
            subsection.paragraphs = paragraphs
            
        except Exception as e:
            log_message(f"Error mapping evidence to subsection {subsection.title}: {e}")
            # Create a placeholder paragraph if mapping fails
            subsection.paragraphs = [
                Paragraph(
                    main_idea=f"Overview of {subsection.title} for {company_name}",
                    points=[SubPoint(content=f"Insufficient evidence available for {company_name}'s {subsection.title} during {time_period}.", sources=["N/A"])],
                    synthesized_content=""
                )
            ]
    
    log_message(f'--- Mapped Evidence to {len(section.subsections)} Subsections for Section: {section.name} ---')
    return {"section": section}


def write_paragraphs(state: SectionState):
    """Write individual paragraphs based on evidence points."""
    section = state["section"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    
    log_message(f'--- Writing Paragraphs for Section: {section.name} ---')

    for subsection in section.subsections:
        for paragraph in subsection.paragraphs:
            evidence_text = "\n".join([
                f"- {point.content} (Source: {', '.join(point.sources)})"
                for point in paragraph.points
            ])
            
            paragraph_prompt = f"""
            You are writing a paragraph for a market performance report on {company_name} during {time_period}.
            
            SECTION: {section.name}
            SUBSECTION: {subsection.title}
            MAIN IDEA: {paragraph.main_idea}
            
            Write a cohesive, well-structured paragraph that synthesizes these evidence points:
            
            {evidence_text}
            
            Guidelines:
            1. This is a Business Strategy Report, not a General Industry Report.
            2. Focus specifically on {company_name}'s performance during {time_period}
            3. Maintain a professional business writing style
            4. Use a logical flow of information by connecting multiple data points to provide deeper insights
            5. Cite specific data points from the evidence where relevant
            6. each paragraph should help readers understand not just what happened, but why it happened
            7. The paragraph should be 150-200 words
            8. Do not include the main idea as a heading or title in your response
            9. Include in-text citations as well as the source link after each fact or statistic in (Author/Organization, Year) format
            10. Ensure every piece of information from the evidence is properly cited
            """
            
            paragraph_content = llm.invoke([
                SystemMessage(content=paragraph_prompt),
                HumanMessage(content=f"Write a paragraph about {paragraph.main_idea} for {company_name}'s {time_period} market analysis.")
            ])
            paragraph.synthesized_content = paragraph_content.content

    log_message(f'--- Wrote Paragraphs for Section: {section.name} ---')
    return {"section": section}


def synthesize_subsections(state: SectionState):
    """Synthesize paragraphs into subsections."""
    section = state["section"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    
    log_message(f'--- Synthesizing Subsections for Section: {section.name} ---')

    for subsection in section.subsections:
        paragraphs_text = "\n\n".join([
            paragraph.synthesized_content for paragraph in subsection.paragraphs
        ])
        
        # Add stronger emphasis on company focus
        company_focus_instruction = f"""
        EXTREMELY IMPORTANT: Your subsection MUST focus SPECIFICALLY on {company_name}. 
        
        DO NOT write generic industry analysis that could apply to any company.
        DO NOT discuss other companies except when directly comparing them to {company_name}.
        
        Every paragraph should mention {company_name} at least once.
        If the input paragraphs contain little specific information about {company_name}, DO NOT add generic filler content.
        Instead, be transparent about the limitations of the available data while maintaining focus on {company_name}.
        """
        
        subsection_prompt = f"""
        You are synthesizing a subsection for a market performance report on {company_name} during {time_period}.
        
        SECTION: {section.name}
        SUBSECTION: {subsection.title}
        
        {company_focus_instruction}
        
        Synthesize these paragraphs into a cohesive subsection:
        
        {paragraphs_text}
        
        Guidelines:
        1. Create a logical flow between paragraphs by connecting multiple data points to provide deeper insights
        2. Add transitions between main ideas
        3. Maintain a professional business writing style
        4. Use proper Markdown formatting
        5. Do not include the subsection title as a heading in your response
        6. Focus on {company_name}'s performance during {time_period}
        7. The subsection should be 400-600 words
        8. Include in-text citations after each fact or statistic in (Author/Organization, Year) format
        9. Ensure every piece of information from the evidence is properly cited
        """
        
        subsection_content = llm.invoke([
            SystemMessage(content=subsection_prompt),
            HumanMessage(content=f"Synthesize the '{subsection.title}' subsection for {company_name}'s {time_period} market analysis.")
        ])
        subsection.synthesized_content = subsection_content.content

    log_message(f'--- Synthesized Subsections for Section: {section.name} ---')
    return {"section": section}


def synthesize_section(state: SectionState):
    """Synthesize subsections into a complete section."""
    section = state["section"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    
    log_message(f'--- Synthesizing Section: {section.name} ---')

    subsections_text = "\n\n".join([
        f"### {subsection.title}\n\n{subsection.synthesized_content}" 
        for subsection in section.subsections
    ])
    
    # Add stronger emphasis on company focus
    company_focus_instruction = f"""
    EXTREMELY IMPORTANT: Your section MUST focus SPECIFICALLY on {company_name}. 
    
    DO NOT write generic industry analysis that could apply to any company.
    DO NOT discuss other companies except when directly comparing them to {company_name}.
    
    The introduction should clearly establish that this section is about {company_name}.
    Every subsection should maintain focus on {company_name}'s specific situation and data.
    """
    
    section_prompt = f"""
    You are synthesizing a complete section for a market performance report on {company_name} during {time_period}.
    
    SECTION: {section.name}
    DESCRIPTION: {section.description}
    
    {company_focus_instruction}
    
    Below are the subsections you need to integrate into a cohesive section:
    
    {subsections_text}
    
    Guidelines:
    1. Create a brief introduction to the section that summarizes key findings
    2. Maintain the existing subsection structure
    3. Ensure logical and cohesive flow between subsections
    4. Use a professional business writing style
    5. Focus on {company_name}'s performance during {time_period}
    6. Use proper Markdown formatting, including headings for each subsection
    7. Do not repeat the main section title in your response
    8. Include in-text citations as well as the source link after each fact or statistic in (Author/Organization, Year) format
    9. Ensure every piece of information from the evidence is properly cited
    """
    
    section_content = llm.invoke([
        SystemMessage(content=section_prompt),
        HumanMessage(content=f"Synthesize the '{section.name}' section for {company_name}'s {time_period} market analysis.")
    ])
    section.content = section_content.content

    log_message(f'--- Synthesized Section: {section.name} ---')
    return {"completed_sections": [section]}


section_builder = StateGraph(SectionState, output=SectionOutputState)

section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("collect_evidence", collect_evidence)
section_builder.add_node("map_evidence_to_subsections", map_evidence_to_subsections)
section_builder.add_node("write_paragraphs", write_paragraphs)
section_builder.add_node("synthesize_subsections", synthesize_subsections)
section_builder.add_node("synthesize_section", synthesize_section)

section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "collect_evidence")
section_builder.add_edge("collect_evidence", "map_evidence_to_subsections")
section_builder.add_edge("map_evidence_to_subsections", "write_paragraphs")
section_builder.add_edge("write_paragraphs", "synthesize_subsections")
section_builder.add_edge("synthesize_subsections", "synthesize_section")
section_builder.add_edge("synthesize_section", END)

section_builder_subagent = section_builder.compile()


#############################################
# FINAL SECTION WRITING FUNCTIONS
#############################################

def format_completed_sections(state: ReportState):
    """Gather completed sections and format them for final context."""
    log_message('--- Formatting Completed Sections ---')
    completed_sections = state["sections"]
    completed_report_sections = format_sections(completed_sections)
    log_message('--- Formatting Completed Sections is Done ---')
    return {"report_sections_from_research": completed_report_sections}


def write_final_sections(state: SectionState):
    """
    Write the final sections of the report, which do not require web search.
    Instead, this synthesizes information from the completed research sections.
    """
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    company_name = state.get("company_name", "")
    time_period = state.get("time_period", "")
    
    log_message(f'--- Writing Final Section: {section.name} ---')
    
    # Load Recess information if this is the recommendations section
    recess_info = ""
    if section.name == "Actionable Recommendations for Recess":
        try:
            recess_info_path = os.path.join("prompts", "recess_info.txt")
            if os.path.exists(recess_info_path):
                with open(recess_info_path, 'r', encoding='utf-8') as f:
                    recess_info = f.read()
                log_message("Loaded Recess information for recommendations section")
            else:
                log_message(f"Warning: Recess info file not found at {recess_info_path}")
        except Exception as e:
            log_message(f"Error loading Recess information: {e}")
    
    # For sections that don't require research (like recommendations),
    # we'll populate each predefined subsection
    for subsection in section.subsections:
        placeholders = {
            "section_title": section.name,
            "section_description": section.description,
            "subsection_title": subsection.title,
            "completed_sections": completed_report_sections,
            "company_name": company_name,
            "time_period": time_period
        }
        
        # Define special handling for specific recommendation subsections
        recess_subsection_guidance = ""
        if section.name == "Actionable Recommendations for Recess":
            recess_subsection_guidance = f"""
            IMPORTANT RECESS INFORMATION:
            {recess_info}
            
            Based on this information about Recess and how they work with CPG brands, provide specific actionable recommendations for how {company_name} could leverage Recess's platform.
            
            For this "{subsection.title}" subsection specifically:
            """
            
            if subsection.title == "Sampling & Activation Opportunities":
                recess_subsection_guidance += f"""
                - Identify specific {company_name} product categories/brands that would benefit most from sampling
                - Recommend specific Recess venue types (coworking, fitness, youth sports, etc.) that align with these products
                - Consider seasonal and promotional timing that would work best for {company_name}
                - Suggest specific activation formats based on {company_name}'s marketing needs identified in earlier sections
                """
            elif subsection.title == "Audience & Channel Recommendations":
                recess_subsection_guidance += f"""
                - Map {company_name}'s target consumer segments to Recess's partner network
                - Identify which Recess venue types best reach {company_name}'s priority demographics
                - Recommend specific geographic targeting based on {company_name}'s regional strengths/weaknesses
                - Suggest audience segmentation strategies for different product lines
                """
            elif subsection.title == "Retailer-Audience Alignment":
                recess_subsection_guidance += f"""
                - Focus heavily on how {company_name} can leverage Recess's partnerships with Dollar General Media Network and the upcoming Walmart Connect partnership
                - Suggest how {company_name} could integrate Recess sampling into their existing retail media investments
                - Recommend specific retailer-focused campaigns to drive in-store sales lift
                - Provide ideas for how {company_name} could use Recess as part of their Joint Business Proposals with retailers
                """
            elif subsection.title == "Scaling Tactics":
                recess_subsection_guidance += f"""
                - Highlight how Recess's model (using existing staff at partner locations) allows {company_name} to scale sampling without the costs of traditional pop-up tours
                - Recommend phased approach to scale sampling across multiple product lines
                - Suggest how {company_name} could test and optimize their Recess campaigns
                - Provide ideas for multi-region or national sampling programs
                """
            elif subsection.title == "Tie-Ins to Company Goals":
                recess_subsection_guidance += f"""
                - Connect Recess's capabilities directly to {company_name}'s stated business objectives from earlier sections
                - Show how Recess sampling can address specific challenges (like underperforming product categories)
                - Recommend how to measure ROI from Recess activities
                - Suggest KPIs that align with {company_name}'s broader marketing goals
                """
        
        subsection_prompt = f"""
        You are writing a subsection for a business report on {company_name}'s market performance during {time_period}.
        
        SECTION: {placeholders['section_title']}
        SUBSECTION: {placeholders['subsection_title']}
        SECTION DESCRIPTION: {placeholders['section_description']}
        COMPANY: {company_name}
        TIME PERIOD: {time_period}
        
        {recess_subsection_guidance}
        
        This is a synthesis section that does NOT require new research. Instead, you should analyze and interpret 
        the information already gathered in previous research sections to provide insights and recommendations.
        
        Here are the completed research sections to draw from:
        
        {placeholders['completed_sections']}
        
        Please write a comprehensive subsection that:
        1. Synthesizes relevant information from the research sections
        2. Provides actionable insights specific to this subsection's topic
        3. Uses a professional business writing style
        4. Is formatted in proper Markdown with appropriate headers, bullet points, etc.
        5. Does not repeat the subsection title (it will be added separately)
        6. Is approximately 250-300 words
        7. Specifically addresses {company_name}'s performance during {time_period}
        
        Focus specifically on the "{placeholders['subsection_title']}" aspect of the section.
        """
        
        # Use LLM to write the subsection content
        llm = get_llm("o3-mini")
        subsection_content = llm.invoke([
            SystemMessage(content=subsection_prompt),
            HumanMessage(content=f"Write the '{subsection.title}' subsection for {company_name}'s {time_period} report.")
        ])
        
        # Create a paragraph with the generated content
        paragraph = Paragraph(
            main_idea=f"Overview of {subsection.title}",
            points=[SubPoint(content="Synthesized from research sections", sources=["Research Sections"])],
            synthesized_content=subsection_content.content
        )
        
        # Add to subsection
        subsection.paragraphs = [paragraph]
        subsection.synthesized_content = subsection_content.content
    
    # Combine all subsections into the final section content
    section_content = f"# {section.name}\n\n"
    for subsection in section.subsections:
        section_content += f"## {subsection.title}\n\n{subsection.synthesized_content}\n\n"
    
    section.content = section_content
    log_message(f'--- Writing Final Section: {section.name} Completed ---')
    return {"completed_sections": [section]}


#############################################
# FINAL REPORT COMPILATION FUNCTIONS
#############################################

def compile_final_report(state: ReportState):
    """
    Compile the final report with a top-down approach
    and gather references from each section's search_docs.
    """
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}
    company_name = state["company_name"]
    time_period = state["time_period"]
    
    log_message('--- Compiling Final Report ---')

    # Replace section content with completed content
    for section in sections:
        section.content = completed_sections.get(section.name, "")

    # Extract key insights from each section
    section_insights = []
    for section in sections:
        if not section.content:
            continue
        insight_prompt = f"""
        Extract 3-5 key insights from this section:
        
        {section.name}:
        {section.content}
        
        Format each insight as a bullet point.
        """
        insights = llm.invoke([
            SystemMessage(content=insight_prompt),
            HumanMessage(content="Extract key insights")
        ]).content
        section_insights.append({
            "section": section.name,
            "insights": insights
        })

    # Load Recess information for context
    recess_info = ""
    try:
        recess_info_path = os.path.join("prompts", "recess_info.txt")
        if os.path.exists(recess_info_path):
            with open(recess_info_path, 'r', encoding='utf-8') as f:
                recess_info = f.read()
            log_message("Loaded Recess information for executive summary context")
    except Exception as e:
        log_message(f"Warning: Could not load Recess information for context: {e}")

    # Build an executive summary
    insights_text = "\n\n".join([
        f"### {item['section']} Insights:\n{item['insights']}"
        for item in section_insights
    ])
    summary_prompt = f"""
    Create an executive summary for a market performance analysis report on {company_name} for {time_period}.
    
    This report was created to identify opportunities for {company_name} to partner with Recess, 
    a turnkey, scalable sampling and brand activation platform.
    
    Use these key insights from each section:
    
    {insights_text}
    
    ABOUT RECESS (Include this context in your summary):
    {recess_info[:500]}... [abbreviated for brevity]
    
    The executive summary should:
    1. Synthesize the most important findings across all sections about {company_name}'s market performance
    2. Highlight key opportunities for {company_name} to leverage Recess's platform based on the findings
    3. Present a coherent overview of the entire report
    4. Be approximately 250-350 words
    5. Follow a professional business style
    6. Be appropriate for marketing executives and business leaders
    7. Specifically mention the company name and time period
    8. Include a brief mention of Recess's capabilities that are most relevant to {company_name}'s needs
    """
    executive_summary = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content="Generate executive summary")
    ]).content

    report_content = f"# {company_name} Market Performance Analysis: {time_period}\n\n## Executive Summary\n\n{executive_summary}\n\n"

    # Build Table of Contents
    toc = "## Table of Contents\n\n"
    for i, section in enumerate(sections, 1):
        if section.content:
            toc += f"{i}. [{section.name}](#section-{i})\n"
    report_content += f"{toc}\n\n"

    # Add sections with anchors and properly formatted Markdown
    for i, section in enumerate(sections, 1):
        if section.content:
            # Create proper heading for the section
            report_content += f"<a id='section-{i}'></a>\n\n## {section.name}\n\n{section.content}\n\n"

    # Now gather references from each section's search_docs
    all_search_docs = []
    for sec in state["completed_sections"]:
        # sec.search_docs holds the raw docs for that section
        if hasattr(sec, "search_docs") and sec.search_docs:
            all_search_docs.extend(sec.search_docs)

    references = build_references(all_search_docs)
    report_content += f"\n{references}\n"

    # Escape unescaped $ symbols
    formatted_report = report_content.replace("\\$", "TEMP_PLACEHOLDER")
    formatted_report = formatted_report.replace("$", "\\$")
    formatted_report = formatted_report.replace("TEMP_PLACEHOLDER", "\\$")

    # Generate filename with company name and time period
    clean_company = re.sub(r'[^\w]', '_', company_name)
    clean_period = re.sub(r'[^\w]', '_', time_period)
    filename = f"{clean_company}_{clean_period}_market_analysis.md"
    file_path = os.path.join(os.getcwd(), filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_report)
        log_message(f"\n--- Report saved as {filename} ---")

    state["final_report"] = formatted_report
    state["filename"] = filename
    return state


#############################################
# PARALLELIZATION FUNCTIONS
#############################################

def parallelize_section_writing(state: ReportState):
    """
    Kick off web research for sections that require it,
    in parallel using the subagent.
    """
    company_name = state["company_name"]
    time_period = state["time_period"]
    company_brands = state.get("company_brands", [])  # Get company brands
    
    return [
        Send("section_builder_with_web_search",
             {"section": s, 
              "company_name": company_name, 
              "time_period": time_period,
              "company_brands": company_brands})  # Pass company_brands to section state
        for s in state["sections"]
        if s.research
    ]


def parallelize_final_section_writing(state: ReportState):
    """
    Write final sections that do not require research, in parallel.
    """
    return [
        Send("write_final_sections",
             {"section": s, 
              "report_sections_from_research": state["report_sections_from_research"],
              "company_name": state["company_name"], 
              "time_period": state["time_period"],
              "company_brands": state.get("company_brands", [])})  # Pass company_brands
        for s in state["sections"]
        if not s.research
    ]


#############################################
# CHATBOT AND AGENT SETUP
#############################################

class ResearchChatbot:
    def __init__(self, agent):
        self.agent = agent
        self.console = Console()

    async def handle_input(self, state: ReportState):
        log_message("ResearchChatbot.handle_input called with state:")
        log_message(f"[bold green]Company: {state['company_name']}[/bold green]")
        log_message(f"[bold green]Time Period: {state['time_period']}[/bold green]")
        log_message(f"Config for report generation: {state.get('config', {})}")
        final_report_response = await call_planner_agent(self.agent, state)
        return final_report_response


async def call_planner_agent(agent, full_state, config={"recursion_limit": 50}):
    console = Console()
    
    # Debug the incoming state
    log_message(f"DEBUG: call_planner_agent received full_state with company_name='{full_state.get('company_name', 'MISSING')}', time_period='{full_state.get('time_period', 'MISSING')}'")
    
    events = agent.astream(
        full_state,
        config,
        stream_mode="values",
    )
    async for event in events:
        # Debug events to see if company_name is preserved
        if isinstance(event, dict) and ('company_name' in event or 'sections' in event):
            log_message(f"DEBUG: Event has company_name='{event.get('company_name', 'MISSING')}'")
            
        if 'final_report' in event:
            md = RichMarkdown(event['final_report'])
            console.print(md)
            return event


async def initialize_agent() -> StateGraph:
    # Create main report builder graph
    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("discover_company_brands", discover_company_brands)  # Add the new node
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("format_completed_sections", format_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)

    builder.add_edge(START, "generate_report_plan")
    builder.add_edge("generate_report_plan", "discover_company_brands")  # Connect plan to brand discovery
    builder.add_conditional_edges("discover_company_brands", parallelize_section_writing, ["section_builder_with_web_search"])  # From brand discovery to section writing
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections", parallelize_final_section_writing, ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    return builder.compile()


#############################################
# MAIN EXECUTION
#############################################


async def main():
    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)
    # Skip clarification step in the graph
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("discover_company_brands", discover_company_brands)  # Add the new node
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("format_completed_sections", format_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)

    # Start directly with generating the report plan
    builder.add_edge(START, "generate_report_plan")
    builder.add_edge("generate_report_plan", "discover_company_brands")  # Connect plan to brand discovery
    builder.add_conditional_edges("discover_company_brands", parallelize_section_writing, ["section_builder_with_web_search"])  # From brand discovery to section writing
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections", parallelize_final_section_writing, ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    reporter_agent = builder.compile()

    chatbot = ResearchChatbot(reporter_agent)

    # Get company name and time period directly from terminal input
    company_name = input("Enter the company name: ")
    if not company_name.strip():
        log_message("Error: Company name cannot be empty.")
        return
    
    time_period = input("Enter the time period (e.g., 'Q4 2023'): ")
    if not time_period.strip():
        log_message("Error: Time period cannot be empty.")
        return
    
    log_message(f"Analyzing {company_name} for {time_period}")
    
    # Set topic as a combination of company_name and time_period for search purposes
    topic = f"{company_name} {time_period} performance"
    # Initial state without clarification fields
    state: ReportState = {
        "company_name": company_name,
        "time_period": time_period,
        "topic": topic,
        "sections": [],
        "completed_sections": [],
        "report_sections_from_research": "",
        "final_report": "",
        "filename": "",
        "company_brands": [],  # Initialize with empty list
        "config": {
            "research_type": "Business",
            "target_audience": "Marketing Executives",
            "writing_style": "Professional"
        }
    }
    
    # Verify the state is correctly initialized
    log_message(f"DEBUG: Initial state company_name='{state['company_name']}', time_period='{state['time_period']}'")
    if not state['company_name']:
        log_message("ERROR: company_name is empty in initial state!")
    if not state['time_period']:
        log_message("ERROR: time_period is empty in initial state!")

    await chatbot.handle_input(state)

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
    asyncio.run(main())