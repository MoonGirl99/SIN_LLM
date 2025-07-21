"""
This file contains the paths to the downloaded pdfs and the extracted text files.
"""

PDF_PATHS = "../downloaded_papers"
OUTPUT_PATH = "../extracted_pdfs"
QUERY = ["ecological networks", "trophic interactions", "network ecology", "food web structure", "species clustering",
"predator-prey dynamics", "mutualistic interactions"]

query = (
    '"species" AND "interaction" AND "ecology"'
)

API_KEY_SPRINGER = ''
API_KEY_ELSEVIER = ''
claude_api = ""
huggingface_api = ""
"""keysearch= ("species" OR "organism" OR "taxa" OR "biodiversity") AND ("interaction" OR "relationship" OR "association" OR "interdependence") AND ("ecology" OR "ecosystem" OR "environment" OR "habitat") AND ("diversity" OR "variety" OR "richness" OR "abundance") AND ("community" OR "population" OR "assemblage" OR "biota")"""