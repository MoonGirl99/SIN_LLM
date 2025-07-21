# species_relationship_extractor.py

import os
import json
import pandas as pd
from tqdm import tqdm
import re
import logging
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeciesRelationshipExtractor:
    """Extract ecological relationships between species from text with NER results."""

    def __init__(self, model_name="meta-llama/Llama-3.3-70B-Instruct", temperature=0.1, batch_size=10, 
                 confidence_threshold=0.5, debug=False):
        self.model_name = model_name
        self.temperature = temperature
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.debug = debug

        self.interaction_df = None
        self.interaction_details = {}
        self.interaction_to_iri = {}

        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompt_template = self._create_few_shot_prompt()

        logger.info(f"Initialized SpeciesRelationshipExtractor with model: {model_name}, temp: {temperature}")

    def _format_relationship_types(self):
        formatted = []
        for interaction, details in self.interaction_details.items():
            if 'source' in details and 'target' in details:
                description = f"{interaction}: {details['source']} → {details['target']}"
            else:
                description = interaction
            formatted.append(f"- {description}")
        return "\n".join(formatted)

    def load_interaction_types(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded interaction types with {len(df)} rows")
            interaction_details = {}
            interaction_to_iri = {}
            for _, row in df.iterrows():
                interaction = row['interaction']
                interaction_details[interaction] = {
                    'source': row['source'],
                    'target': row['target'],
                    'termIRI': row['termIRI']
                }
                if 'termIRI' in row:
                    interaction_to_iri[interaction] = row['termIRI']
            self.interaction_df = df
            self.interaction_details = interaction_details
            self.interaction_to_iri = interaction_to_iri
            return df, interaction_details, interaction_to_iri
        except Exception as e:
            logger.error(f"Error loading interaction types: {str(e)}")
            return None, {}, {}

    def _create_few_shot_prompt(self):
        template = """
You are an expert in extracting ecological relationships between species entities.  
Your task is to identify whether there is a specific interaction between the annotated entities in the given text.

Only use the following interaction types, grouped by category:

Predation:  
- eats, eatenby, kills, killedby, preyson, preyeduponby  

Competition:  
- allelopathof, competeswith  

Mutualism:  
- pollinates, pollinatedby, mutualistof, symbiontof, providesnutrientsfor, acquiresnutrientsfrom  

Commensalism:  
- commensalistof, epiphyteof, hasepiphyte, flowersvisitedby, visitsflowersof, dispersalvectorof, hasdispersalvector  

Parasitism:  
- parasiteof, hasparasite, hostof, hashost, parasitoidof, hasparasitoid, endoparasiteof, hasendoparasite, pathogenof, haspathogen, ectoparasiteof, hasectoparasite  

Neutral:  
- ecologicallyrelatedto, cooccurswith, coroostswith, interactswith, adjacentto, hashabitat, createshabitatfor, vectorof, hasvector  

Use `interactswith` **only** when a relation is likely but not explicitly stated (e.g., co-occurrence in ecological context).

If no specific relation is mentioned, do not assign one.

---

### Examples

**Example 1 (Predation):**  
Text: "Brown trout (Salmo trutta) prey on native roundhead galaxias (Galaxias anomalus)."  
E1: Brown trout  
E2: roundhead galaxias  
relation: E1 preyson E2  
confidence: 0.9

**Example 2 (Parasitism):**  
Text: "The parasitic mite Varroa destructor has devastated honeybee (Apis mellifera) colonies."  
E1: Varroa destructor  
E2: Apis mellifera  
relation: E1 parasiteof E2  
confidence: 0.9

**Example 3 (Competition):**  
Text: "Oak trees (Quercus spp.) compete with maple trees (Acer spp.) for light."  
E1: Oak trees  
E2: maple trees  
relation: E1 competeswith E2  
confidence: 0.9

**Example 4 (Neutral / Fallback):**  
Text: "Barnacles are often found on whale skin."  
E1: barnacles  
E2: whales  
relation: E1 interactswith E2  
confidence: 0.8

**Example 5 (Negative - No relation):**  
Text: "Bald eagles and pine trees are found in the same national parks."  
E1: Bald eagles  
E2: pine trees  
relation: (No relationship detected.)

**Example 6 (Negative - Generic co-occurrence):**  
Text: "Blue whales and dolphins are both marine animals."  
E1: Blue whales  
E2: dolphins  
relation: (No relationship detected.)

**Example 7 (Predation):**  
Text: "Caterpillars of monarch butterflies (Danaus plexippus) feed on milkweed (Asclepias spp.)."  
E1: Danaus plexippus  
E2: Asclepias  
relation: E1 preyson E2  
confidence: 0.9

---

### Now apply to the following:

Text: {text}  
Entities:  
{entities}

Return only pairs with a clear interaction.  
For each, give:

E1: [entity 1]  
E2: [entity 2]  
relation: E1 [relation type] E2  
confidence: [0.0–1.0]

If there is no relationship, do not include that pair.

Do **not** create new entities or relations.  
Use only the given entity list and approved relation types.

"""
        return PromptTemplate(input_variables=["text", "entities", "interaction_types"], template=template)

    def _passes_entity_filter(self, text):
        return True

    def prepare_document(self, doc):
        filtered_entities = []
        for i, entity in enumerate(doc.get("entities", [])):
            new_entity = entity.copy()
            new_entity["id"] = len(filtered_entities)
            filtered_entities.append(new_entity)
        entity_list = "\n".join([
            f"Entity ID {e['id']}: \"{e['text']}\" (Type: {e.get('label', 'SPECIES')})" 
            for e in filtered_entities
        ])
        return {
            "document_id": doc.get("document_id", "unknown"),
            "text": doc.get("text", ""),
            "original_entities": doc.get("entities", []),
            "filtered_entities": filtered_entities,
            "entity_list": entity_list
        }

    def extract_relationships(self, prepared_doc):
        if len(prepared_doc["filtered_entities"]) < 2:
            return []
        doc_id = prepared_doc.get("document_id", "unknown")
        prompt_inputs = {
            "interaction_types": self._format_relationship_types(),
            "text": prepared_doc["text"],
            "entities": prepared_doc["entity_list"]
        }
        prompt = self.prompt_template.format(**prompt_inputs)
        try:
            response = self.llm.invoke(prompt)
            response_content = getattr(response, 'content', '')
            if self.debug:
                logger.info(f"Response preview: {response_content[:200]}")
            relationships = []
            relationship_pattern = r'E1:\s*([^\n]+)\s*\nE2:\s*([^\n]+)\s*\nrelation:\s*E1\s+(\w+)\s+E2\s*\nconfidence:\s*(0\.\d+|1\.0)'
            matches = re.findall(relationship_pattern, response_content)
            seen = set()
            for match in matches:
                source, target, rel, conf = map(str.strip, match)
                key = (source.lower(), target.lower(), rel)
                if key not in seen and float(conf) >= self.confidence_threshold:
                    seen.add(key)
                    relationships.append({
                        "source_text": source,
                        "target_text": target,
                        "relation": rel,
                        "confidence": float(conf)
                    })
            return relationships
        except Exception as e:
            logger.error(f"Error extracting relationships for doc {doc_id}: {str(e)}")
            return []

    def load_ner_results(self, ner_results_path):
        try:
            df = pd.read_csv(ner_results_path)
            documents = []
            for doc_id, group in df.groupby('doc_id'):
                content = group['content'].iloc[0] if 'content' in group.columns else ""
                entities = []
                for _, row in group.iterrows():
                    if pd.notna(row.get('start')) and pd.notna(row.get('end')) and pd.notna(row.get('Annotation Text')):
                        entity_text = row['Annotation Text']
                        if self._passes_entity_filter(entity_text):
                            entities.append({
                                'id': len(entities),
                                'text': entity_text,
                                'start': int(row['start']),
                                'end': int(row['end']),
                                'label': row.get('label', 'SPECIES')
                            })
                if entities:
                    documents.append({
                        'document_id': doc_id,
                        'text': content,
                        'entities': entities
                    })
            return documents
        except Exception as e:
            logger.error(f"Error loading NER results: {str(e)}")
            return []

    def process_documents(self, documents, output_path=None):
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            for doc in tqdm(batch):
                prepared_doc = self.prepare_document(doc)
                relationships = self.extract_relationships(prepared_doc)
                doc["relationships"] = relationships
                results.append(doc)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        return results

    def run_pipeline(self, ner_results_path, output_path):
        documents = self.load_ner_results(ner_results_path)
        return self.process_documents(documents, output_path) 