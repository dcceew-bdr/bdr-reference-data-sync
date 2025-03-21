import pathlib
import logging
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, DCTERMS, Namespace, SKOS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define namespaces
PROF = Namespace("http://www.w3.org/ns/dx/prof/")
SCHEMA = Namespace("https://schema.org/")
MRR = Namespace("https://prez.dev/ManifestResourceRoles/")
DCAT = Namespace("http://www.w3.org/ns/dcat#")


def extract_name(g, subject):
    """
    Extract a name for the entity from dcterms:title, rdfs:label, or skos:prefLabel.
    Returns None if no name is found.
    """
    title = list(g.objects(subject, DCTERMS.title))
    if title:
        return title[0]

    label = list(g.objects(subject, RDFS.label))
    if label:
        return label[0]

    pref_label = list(g.objects(subject, SKOS.prefLabel))
    if pref_label:
        return pref_label[0]

    return None


def extract_description(g, subject):
    """
    Extract a description for the entity from dcterms:description.
    Returns None if no description is found.
    """
    description = list(g.objects(subject, DCTERMS.description))
    if description:
        return description[0]

    return None


def find_main_entity_catalog(file_path):
    """
    Find the main entity in a catalog.ttl file.
    The main entity is the one with rdf:type dcat:Catalog.
    Returns the URI of the main entity or None if not found.
    Logs a concise message about the entity found.
    """
    g = Graph()
    g.parse(file_path, format="turtle")
    filename = pathlib.Path(file_path).name
    parent_dir = pathlib.Path(file_path).parent.name

    catalog_entities = set(g.subjects(RDF.type, DCAT.Catalog))

    if len(catalog_entities) == 0:
        logger.error(f"No dcat:Catalog found in {parent_dir}/{filename}")
        return None, None
    elif len(catalog_entities) > 1:
        logger.error(f"Multiple dcat:Catalog entities found in {parent_dir}/{filename}")
        return None, None

    entity = next(iter(catalog_entities))
    logger.info(f"Main entity of {parent_dir}/{filename} is {entity} (dcat:Catalog)")
    return entity, g


def find_main_entity_vocabulary(file_path):
    """
    Find the main entity in a vocabulary TTL file.
    The main entity is the one with rdf:type skos:ConceptScheme.
    If no skos:ConceptScheme is found, falls back to finding the subject of skos:hasTopConcept.
    Returns the URI of the main entity or None if not found.
    Uses set operation to handle duplicate entries.
    """
    g = Graph()
    g.parse(file_path, format="turtle")
    filename = pathlib.Path(file_path).name

    # Try to find entities with rdf:type skos:ConceptScheme
    concept_schemes = set(g.subjects(RDF.type, SKOS.ConceptScheme))

    if len(concept_schemes) == 1:
        entity = next(iter(concept_schemes))
        logger.info(f"Main entity of {filename} is {entity} (skos:ConceptScheme)")
        return entity, g
    elif len(concept_schemes) > 1:
        logger.error(f"Multiple skos:ConceptScheme entities found in {filename}")
        return None, None

    # If no concept scheme found, try fallback: subject of skos:hasTopConcept
    logger.info(f"No skos:ConceptScheme found in {filename}, trying fallback...")
    top_concept_subjects = set(g.subjects(SKOS.hasTopConcept, None))

    if len(top_concept_subjects) == 0:
        logger.error(f"No skos:hasTopConcept predicates found in {filename}")
        return None, None
    elif len(top_concept_subjects) == 1:
        entity = next(iter(top_concept_subjects))
        logger.info(f"Main entity of {filename} is {entity} (has skos:hasTopConcept)")
        return entity, g
    else:
        # If we have multiple entries but they're all the same URI (just duplicated)
        unique_uris = {str(uri) for uri in top_concept_subjects}
        if len(unique_uris) == 1:
            entity = next(iter(top_concept_subjects))
            logger.info(f"Main entity of {filename} is {entity} (has skos:hasTopConcept)")
            return entity, g
        else:
            logger.error(f"Multiple distinct subjects with skos:hasTopConcept found in {filename}")
            return None, None


def generate_prez_manifest(root_dir, output_file):
    """
    Generate a Prez manifest based on the directory structure provided.
    Each catalog.ttl will be marked with mrr:CatalogueData role.
    Each file under vocabularies will be marked with mrr:ResourceData role.
    Finds and records the main entity for each file.
    """
    g = Graph()
    root_path = pathlib.Path(root_dir)

    # Bind namespaces
    g.bind("prof", PROF)
    g.bind("schema", SCHEMA)
    g.bind("mrr", MRR)
    g.bind("dcterms", DCTERMS)
    g.bind("dcat", DCAT)
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("prez", Namespace("https://prez.dev/"))
    g.bind("xsd", XSD)

    # Create the manifest node
    manifest = URIRef("https://example.org/profile/my-prez-manifest")
    g.add((manifest, RDF.type, URIRef("https://prez.dev/Manifest")))

    # Process directories to find catalog.ttl and vocabulary files
    for dirpath in root_path.glob("**/"):
        # Check for catalog.ttl
        catalog_file = dirpath / "catalog.ttl"
        if catalog_file.exists():
            rel_path = str(catalog_file.relative_to(root_path))

            # Find the main entity in the catalog file
            main_entity, graph = find_main_entity_catalog(catalog_file)

            if main_entity:
                # Get the name and description from the main entity
                name = extract_name(graph, main_entity)
                description = extract_description(graph, main_entity)

                # Create a resource for the catalog.ttl
                catalog_resource = BNode()
                g.add((manifest, PROF.hasResource, catalog_resource))

                # For catalogs, use a simple string literal for the artifact
                g.add((catalog_resource, PROF.hasArtifact, Literal(rel_path)))

                # Add remaining properties
                g.add((catalog_resource, PROF.hasRole, MRR.CatalogueData))

                # Add name and description if available
                if name:
                    g.add((catalog_resource, SCHEMA.name, name))
                if description:
                    g.add((catalog_resource, SCHEMA.description, description))

        # Check for vocabulary files under 'vocabularies' directory
        vocab_dir = dirpath / "vocabularies"
        if vocab_dir.exists() and vocab_dir.is_dir():
            for vocab_file in vocab_dir.glob("*.ttl"):
                rel_path = str(vocab_file.relative_to(root_path))

                # Find the main entity in the vocabulary file
                main_entity, graph = find_main_entity_vocabulary(vocab_file)

                if main_entity:
                    # Get the name and description from the main entity
                    name = extract_name(graph, main_entity)
                    description = extract_description(graph, main_entity)

                    # Create a resource for the vocabulary file
                    vocab_resource = BNode()
                    g.add((manifest, PROF.hasResource, vocab_resource))

                    # Create a blank node for the artifact with mainEntity
                    artifact_bn = BNode()
                    g.add((vocab_resource, PROF.hasArtifact, artifact_bn))
                    g.add((artifact_bn, SCHEMA.contentLocation, Literal(rel_path)))
                    g.add((artifact_bn, SCHEMA.mainEntity, main_entity))

                    # Add role property
                    g.add((vocab_resource, PROF.hasRole, MRR.ResourceData))

                    # Add name and description if available
                    if name:
                        g.add((vocab_resource, SCHEMA.name, name))
                    if description:
                        g.add((vocab_resource, SCHEMA.description, description))

    # Serialize to Turtle
    g.serialize(destination=output_file, format="turtle")
    logger.info(f"Manifest successfully written to {output_file}")


if __name__ == "__main__":
    # Configure the root directory and output file name
    ROOT_DIR = "generated"  # Change this to match your actual root directory if different
    OUTPUT_FILE = "prez_manifest.ttl"

    generate_prez_manifest(ROOT_DIR, OUTPUT_FILE)