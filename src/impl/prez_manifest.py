
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings
import rdflib
from rdflib.namespace import Namespace, PROF, SDO, DCTERMS, RDF

from ..voc_graph import make_voc_graph

SCHEMA = SDO 
PREZ = Namespace("https://prez.dev/")
MRR = Namespace("https://prez.dev/ManifestResourceRoles/")
RDF_TYPE = RDF.type
PROF_HAS_RESOURCE = PROF.hasResource
PROF_HAS_ROLE = PROF.hasRole
PROF_HAS_ARTIFACT = PROF.hasArtifact

# :CompleteCatalogueAndResourceLabels
#     a skos:Concept ;
#     skos:definition "All the labels - possibly including names, descriptions & seeAlso links - for the Catalogue and Resource objects" ;
#     skos:inScheme cs: ;
#     skos:prefLabel "Complete Catalogue and Resource Labels" ;
# .

# :CatalogueAndResourceModel
#     a skos:Concept ;
#     skos:definition "The default model for the container and the content. Must be a set of SHACL Shapes" ;
#     skos:inScheme cs: ;
#     skos:prefLabel "Catalogue & Resource Model" ;
# .

# :CatalogueData
#     a skos:Concept ;
#     skos:definition "Data for the container, usually a Catalogue, including the identity of it and each item fo content" ;
#     skos:inScheme cs: ;
#     skos:prefLabel "Catalogue Data" ;
# .

# :CatalogueModel
#     a skos:Concept ;
#     skos:definition "The default model for the container. Must be a set of SHACL Shapes" ;
#     skos:inScheme cs: ;
#     skos:prefLabel "Catalogue Model" ;
# .

# :ResourceData
#     a skos:Concept ;
#     skos:definition "Data for the content of the container" ;
#     skos:inScheme cs: ;
#     skos:prefLabel "Resource Data" ;
# .

# :ResourceModel
#     a skos:Concept ;
#     skos:definition "The default model for the content. Must be a set of SHACL Shapes" ;
#     skos:inScheme cs: ;
#     skos:prefLabel "Resource Model" ;
# .

# :IncompleteCatalogueAndResourceLabels
#     a skos:Concept ;
#     skos:definition "Some of the labels - possibly including names, descriptions & seeAlso links - for the Catalogue and Resource objects" ;
#     skos:inScheme cs: ;
#     skos:prefLabel "Incomplete Catalogue and Resource Labels" ;
# .

class ManifestResourceRole(Enum):
    CompleteCatalogueAndResourceLabels = MRR.CompleteCatalogueAndResourceLabels
    IncompleteCatalogueAndResourceLabels = MRR.IncompleteCatalogueAndResourceLabels
    CatalogueAndResourceModel = MRR.CatalogueAndResourceModel
    CatalogueData = MRR.CatalogueData
    CatalogueModel = MRR.CatalogueModel
    ResourceData = MRR.ResourceData
    ResourceModel = MRR.ResourceModel

@dataclass
class ManifestResource:
    conforms_to: str|None
    role: ManifestResourceRole
    artifacts: list[str]


@dataclass
class PrezManifest:
    resources: list[ManifestResource]

def load_manifest_from_graph(g: rdflib.Graph) -> PrezManifest:
    # Only works from local files, not remote SPARQL endpoints.
    manifests = list(g.subjects(RDF_TYPE, PREZ.Manifest))
    if len(manifests) == 0:
        raise ValueError("No manifests found in the provided graph.")
    elif len(manifests) > 1:
        raise ValueError("Multiple manifests found in the provided graph. Please provide a graph with only one manifest.")
    manifest_node = manifests[0]
    resource_nodes = list(g.objects(manifest_node, PROF_HAS_RESOURCE))
    resources: list[ManifestResource] = []
    for resource_node in resource_nodes:
        conforms_tos = list(g.objects(resource_node, DCTERMS.conformsTo))
        if len(conforms_tos) > 0:
            conforms_to = str(conforms_tos[0])
        else:
            conforms_to = None
        has_roles = list(g.objects(resource_node, PROF_HAS_ROLE))
        if len(has_roles) == 0:
            # No resource role, skip this resource.
            continue  # TODO: raise an error?
        elif len(has_roles) > 1:
            # Multiple roles, skip this resource.
            continue  # TODO: raise an error?
        artifact_nodes = list(g.objects(resource_node, PROF_HAS_ARTIFACT))
        if len(artifact_nodes) == 0:
            # No artifacts, skip this resource.
            continue  # TODO: raise an error?
        artifacts = [str(a) for a in artifact_nodes]
        resource = ManifestResource(conforms_to, ManifestResourceRole(has_roles[0]), artifacts)
        resources.append(resource)
    return PrezManifest(resources)

def load_manifest_file(manifest_file: Path) -> PrezManifest:
    g = rdflib.Graph(bind_namespaces="core")
    g.parse(manifest_file, format="turtle")
    return load_manifest_from_graph(g)

def is_rdf_file(file_path: Path) -> bool:
    return file_path.suffix in [".ttl", ".turtle", ".nq", ".nquads", ".nt", ".ntriples", ".trig", ".rdf", ".xml", ".jsonld", ".json-ld"]

def guess_rdf_file_format(file_path: Path) -> str:
    if file_path.suffix in [".ttl", ".turtle"]:
        return "turtle"
    elif file_path.suffix in [".nq", ".nquads"]:
        return "nquads"
    elif file_path.suffix in [".nt", ".ntriples"]:
        return "ntriples"
    elif file_path.suffix == ".trig":
        return "trig"
    elif file_path.suffix in [".rdf", ".xml"]:
        return "xml"
    elif file_path.suffix in [".jsonld", ".json-ld"]:
        return "json-ld"
    else:
        raise ValueError(f"Unknown RDF file format for file: {file_path}")

def load_rdf_resources_into_graph(manifest: PrezManifest, base_path: Path|None, into_graph: rdflib.Graph|None) -> rdflib.Graph:
    if into_graph is None:
        g = make_voc_graph()
    else:
        g = into_graph

    if base_path is None:
        base_path = Path(".")

    for resource in manifest.resources:
        role = resource.role
        # TODO: Check the role if its one that we want to load into the graph.
        for artifact in resource.artifacts:
            rdf_file_artifacts: list[Path]
            if "*" in artifact:
                # This is a glob pattern, we need to expand it
                globbed_artifact_files = list(base_path.glob(artifact))
                if len(globbed_artifact_files) == 0:
                    warnings.warn(f"No files found matching glob pattern: {artifact}")
                rdf_file_artifacts = [f for f in globbed_artifact_files if is_rdf_file(f)]
            else:    
                artifact_path = Path(artifact)
                if not artifact_path.is_absolute():
                    artifact_path = base_path / artifact_path
                if not is_rdf_file(artifact_path):
                    # Not an RDF file, skip it
                    continue
                if not artifact_path.exists():
                    raise FileNotFoundError(f"Artifact file {artifact_path} does not exist.")
                rdf_file_artifacts = [artifact_path]
            for artifact_path in rdf_file_artifacts:
                rdf_format = guess_rdf_file_format(artifact_path)
                if rdf_format == "nquads" or rdf_format == "trig":
                    raise NotImplementedError("N-Quads and TriG formats are not supported for loading into a catalogue graph.")
                g.parse(artifact_path, format=rdf_format)
    return g 