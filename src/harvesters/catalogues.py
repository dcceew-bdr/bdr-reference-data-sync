from typing import Any, Optional
import rdflib

from .base import BaseHarvester, SPARQLBaseHarvester, LocalBaseHarvester
from .vocabularies import SPARQLVocabHarvester, LocalVocabHarvester, VocabHarvester
from ..voc_graph import make_voc_graph

class CatalogueHarvester(BaseHarvester):

    def __init__(self, source_graph: rdflib.Graph):
        super().__init__(source_graph)

    async def async_init(self):
        await super().async_init()
        self.is_init = True

    @classmethod
    def build_from_source(cls, source: str, extra_options: dict[str, Any]) -> 'CatalogueHarvester':
        source_lower = source.lower()
        if source_lower.startswith("sparql:"):
            klass = SPARQLCatalogueHarvester
        elif source_lower.startswith("http:") or source_lower.startswith("https:"):
            klass = LocalCatalogueHarvester
        elif source_lower.startswith("file:"):
            klass = LocalCatalogueHarvester
        else:
            raise NotImplementedError(f"Unsupported catalogue source type: {source}")
        return super().build_from_source(source, klass, extra_options)

    def load_def(self, harvester_def: dict[str, Any]):
        super().load_def(harvester_def)

    def make_vocab_harvester(self) -> VocabHarvester:
        raise NotImplementedError("This method should be implemented in subclasses to create a VocabHarvester instance.")
    
    async def harvest_catalogue_details(self, into_graph: Optional[rdflib.Graph] = None, passthrough: bool = False) -> rdflib.Graph:
        if passthrough:
            # The whole source graph is passed through to the output graph
            # Be careful with this, as it may include a lot of data
            # could take a very long time and use a lot of memory
            if into_graph is None:
                return self.source_graph
            else:
                for (s, p, o) in self.source_graph:
                    into_graph.add((s, p, o))
                return into_graph
        
        if into_graph is None:
            g = make_voc_graph()
        else:
            g = into_graph
        if not self.is_init:
            await self.async_init()
        elif self.root_node_details is None and self.root_node is not None:
            self.root_node_details = await self.cbd(self.root_node)

        # TODO: Implement proper catalogue harvesting logic here (Schema:DataCatalog, dcat:Catalog, etc.)

        return g



class SPARQLCatalogueHarvester(SPARQLBaseHarvester, CatalogueHarvester):
    def __init__(self, source_graph: rdflib.Graph, is_graph_db: bool = False):
        super().__init__(source_graph, is_graph_db=is_graph_db)

    def make_vocab_harvester(self) -> SPARQLVocabHarvester:
        new_harvester = SPARQLVocabHarvester(self.source_graph, is_graph_db=self.is_graph_db)
        new_harvester.is_init = False
        new_harvester.name = self.name
        new_harvester.root_node = self.root_node
        new_harvester.graph_name = self.graph_name
        new_harvester.root_node_details = self.root_node_details
        return new_harvester
    
class LocalCatalogueHarvester(LocalBaseHarvester, CatalogueHarvester):
    def __init__(self, source_graph: rdflib.Graph):
        super().__init__(source_graph)
    
    def make_vocab_harvester(self) -> LocalVocabHarvester:
        new_harvester = LocalVocabHarvester(self.source_graph)
        new_harvester.is_init = False
        new_harvester.name = self.name
        new_harvester.root_node = self.root_node
        new_harvester.graph_name = self.graph_name
        new_harvester.root_node_details = self.root_node_details
        return new_harvester