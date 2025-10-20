
from pathlib import Path
from typing import Any, Optional, Type
import httpx
import rdflib
from rdflib.plugins.stores import sparqlstore

from ..impl.prez_manifest import load_manifest_from_graph, load_rdf_resources_into_graph
from ..voc_graph import make_voc_graph
from .sparql_fetch import sparql_describe, sparql_subjects, sparql_objects

class BaseHarvester:
    source_graph: rdflib.Graph
    name: str
    is_init: bool
    root_node: rdflib.URIRef
    root_node_details: Optional[rdflib.Graph]  # CBD of root_node
    graph_name: Optional[rdflib.URIRef]
    
    def __init__(self, source_graph: rdflib.Graph):
        self.is_init = False
        self.source_graph = source_graph
        self.graph_name = None
        self.root_node_details = None


    async def async_init(self):
        pass

    def load_def(self, harvester_def: dict[str, Any]):
        _def_root_node = harvester_def.get("root_node")
        if not _def_root_node:
            root_node = None
        else:
            root_node = rdflib.URIRef(_def_root_node)
        try:
            name = harvester_def["name"]
        except LookupError:
            token = harvester_def.get("token", None)
            if token is None:
                raise RuntimeError("No name or token identified for the Vocabulary definition.")
            else:
                name = token
        if not name:
            raise RuntimeError("No name identified for the Vocabulary definition.")
        self.root_node = root_node  # Todo: Should this be allowed to be None
        self.name = name
        graph_name_str: Optional[str] = harvester_def.get("graph_name", None)
        self.graph_name = rdflib.URIRef(graph_name_str) if graph_name_str else None
        

    @classmethod
    def build_from_source(cls, source: str, klass: Type['BaseHarvester'], extra_options: dict[str, Any]) -> 'BaseHarvester':
        source_lower = source.lower()
        if source_lower.startswith("sparql:"):
            store = sparqlstore.SPARQLStore(query_endpoint=source[7:], method='POST', returnFormat='json')
            source_graph = rdflib.Graph(store=store, bind_namespaces="core")  # bind only core, so SPARQL prefixes in the sparql queries work properly
            is_graph_db = extra_options.get("is_graph_db", None) # GraphDB-specific harvester behaviour
            if is_graph_db is not None:
                vocab_harvester = klass(source_graph, is_graph_db=bool(is_graph_db))
            else:
                vocab_harvester = klass(source_graph)
        elif source_lower.startswith("https:") or source_lower.startswith("http:"):
            source_graph = rdflib.Graph(bind_namespaces="core")  # bind only core, so SPARQL prefixes in the sparql queries work properly
            with httpx.Client() as client:
                try:
                    resp = client.get(source, headers={'Accept': 'text/turtle'}, follow_redirects=True)
                    resp.raise_for_status()
                except Exception as e:
                    print(e)
                    raise
                source_graph.parse(resp.read())
            vocab_harvester = klass(source_graph)
        elif source_lower.startswith("file:"):
            file_uri = source[5:]
            if file_uri.startswith("//"):
                # This is the file:// protocol, note, it _cannot_ be relative
                file_uri = file_uri[2:]
                if "/" not in file_uri:
                    raise RuntimeError(f"File URI {file_uri} is not <host>/<path>")
                host, path = file_uri.split("/", 1)
                if len(host) == 0 or host.lower() == "localhost":
                    # This is a local file
                    local_path = Path(path)
                else:
                    # This is a remote file
                    raise NotImplementedError(f"Remote file URIs are not supported: {file_uri}")
            else:
                # a "file:" string, this is relative to the current working directory
                local_path = Path(".") / Path(file_uri)
            source_graph = make_voc_graph()
            if not local_path.exists():
                raise RuntimeError(f"File {source} does not exist.")
            with open(local_path, "rb") as f:
                source_graph.parse(f)
            root_node = extra_options.get("root_node", None)
            if local_path.name == "manifest.ttl" or root_node == "https://prez.dev/Manifest":
                # this is a prez manifest.
                prez_manifest = load_manifest_from_graph(source_graph)
                orig_graph = source_graph
                source_graph = make_voc_graph()
                load_rdf_resources_into_graph(prez_manifest, local_path.parent, source_graph)
            vocab_harvester = klass(source_graph)
        else:
            raise NotImplementedError(f"Unsupported vocab source type: {source}")
        return vocab_harvester
    
    async def cbd(self, identifier: rdflib.URIRef) -> rdflib.Graph:
        raise NotImplementedError()

    async def subjects(self, p, o):
        raise NotImplementedError()

    async def objects(self, s, p):
        raise NotImplementedError()
    
class SPARQLBaseHarvester(BaseHarvester):
    def __init__(self, source_graph: rdflib.Graph, is_graph_db: bool = False):
        super().__init__(source_graph)
        self.is_graph_db: bool = is_graph_db

    async def cbd(self, identifier: rdflib.URIRef) -> rdflib.Graph:
        return await sparql_describe(self.source_graph, identifier, explicit=self.is_graph_db)

    async def subjects(self, p, o):
        return await sparql_subjects(self.source_graph, p, o, explicit=self.is_graph_db)

    async def objects(self, s, p):
        return await sparql_objects(self.source_graph, s, p, explicit=self.is_graph_db)

class LocalBaseHarvester(BaseHarvester):
    def __init__(self, source_graph: rdflib.Graph):
        super().__init__(source_graph)

    async def cbd(self, identifier: rdflib.URIRef) -> rdflib.Graph:
        return self.source_graph.cbd(identifier)

    async def subjects(self, p, o):
        return self.source_graph.subjects(p, o)

    async def objects(self, s, p):
        return self.source_graph.objects(s, p)