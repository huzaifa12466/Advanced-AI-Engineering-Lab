"""
data_generation.py
──────────────────
Synthetic FinTech data generation using LLM + Instructor.
Generates companies, CEOs, and their relationships.
"""

from typing import List
from src.models import CompaniesList, CompanyDetails
from src.config import LLM_MODEL, NUM_COMPANIES, BATCH_SIZE


def generate_companies_list(instructor_client, n: int = NUM_COMPANIES) -> CompaniesList:
    """
    Step 1: Generate N FinTech company names and their sectors.

    Args:
        instructor_client: Instructor-patched Groq client
        n:                 Number of companies to generate

    Returns:
        CompaniesList with company names and sectors
    """
    return instructor_client.chat.completions.create(
        response_model=CompaniesList,
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} unique fictional FinTech company names across diverse sectors "
                "(Payments, Banking, Insurance, Lending, Crypto, FMCG, Telecom). "
                "Return exactly matching companies and sectors lists."
            )
        }]
    )


def generate_company_details(
    instructor_client,
    batch_names:   List[str],
    batch_sectors: List[str]
) -> List[CompanyDetails]:
    """
    Step 2: Generate detailed profiles for a batch of companies.

    Args:
        instructor_client: Instructor-patched Groq client
        batch_names:       List of company names
        batch_sectors:     Corresponding sectors

    Returns:
        List of CompanyDetails
    """
    details = []
    for company, sector in zip(batch_names, batch_sectors):
        try:
            detail = instructor_client.chat.completions.create(
                response_model=CompanyDetails,
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate a realistic FinTech company profile for '{company}' "
                        f"in the '{sector}' sector. Include CEO background, financial "
                        "performance (profit/loss in millions), and 3-5 business "
                        "relationships with impact percentages and relationship types."
                    )
                }]
            )
            details.append(detail)
            print(f"  ✅ Generated: {company}")
        except Exception as e:
            print(f"  ❌ Failed: {company} — {e}")
    return details


def run_data_generation(instructor_client) -> List[CompanyDetails]:
    """
    Full data generation pipeline.
    Generates company list then fetches details in batches.

    Args:
        instructor_client: Instructor-patched Groq client

    Returns:
        List of all CompanyDetails
    """
    print("🔄 Step 1: Generating company names...")
    companies_list = generate_companies_list(instructor_client)
    print(f"✅ Generated {len(companies_list.companies)} companies\n")

    print("🔄 Step 2: Generating company details in batches...")
    all_companies = []

    for i in range(0, len(companies_list.companies), BATCH_SIZE):
        batch_names   = companies_list.companies[i:i + BATCH_SIZE]
        batch_sectors = companies_list.sectors[i:i + BATCH_SIZE]
        print(f"\n  Batch {i // BATCH_SIZE + 1}: {batch_names}")
        batch_details = generate_company_details(instructor_client, batch_names, batch_sectors)
        all_companies.extend(batch_details)

    print(f"\n✅ Total companies generated: {len(all_companies)}")
    return all_companies
