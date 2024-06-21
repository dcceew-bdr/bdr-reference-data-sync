from typing import List, Dict, Any, Union, Tuple
import rdflib
from rdflib import RDF, DCAT, RDFS
from rdflib.plugins.stores import sparqlstore
from pathlib import Path
from .harvesters import VocabHarvester
from .voc_graph import make_voc_graph, VocabGraphDetails


async def build_catalog(catalog_def: Dict[str, Any]) -> rdflib.Graph:
    try:
        cat_source = catalog_def["source"]
    except LookupError:
        raise RuntimeError("No source defined on catalog definition.")

    harvesters = {}
    harvesters[cat_source] = VocabHarvester.build_from_source(cat_source)
    cat_token = catalog_def["token"]
    cat_path = Path(".") / "generated" / cat_token
    cat_path.mkdir(exist_ok=True, parents=True)
    cat_graph = make_voc_graph()
    cat_uri = rdflib.URIRef(f"https://linked.data.gov.au/dataset/bdr/catalog/{cat_token}")
    cat_graph.add((cat_uri, RDF.type, DCAT.Catalog))
    cat_graph.add((cat_uri, DCAT.themeTaxonomy, rdflib.URIRef("https://linked.data.gov.au/def/abis/themes")))
    vocabularies = catalog_def.get("vocabularies", [])
    vocab_graph_details: List[VocabGraphDetails] = []
    for vocab_def in vocabularies:
        vocab_source = vocab_def.get("source", None)
        if vocab_source:
            if vocab_source in harvesters:
                harvester = harvesters[vocab_source]
            else:
                harvester = VocabHarvester.build_from_source(vocab_source)
        else:
            harvester = harvesters[cat_source]
        harvester.load_def(vocab_def)
        these_vocab_graphs_details: List[VocabGraphDetails] = await harvester.run_procedures()
        vocab_graph_details.extend(these_vocab_graphs_details)
    for vocab_graph_detail in vocab_graph_details:
        out_dir = cat_path / "vocabularies"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / f"{vocab_graph_detail.token}.ttl"
        with open(out_file, "wb") as f:
            vocab_graph_detail.graph.serialize(f, format="turtle")
        vocab_uri = vocab_graph_detail.vocab_uri
        cat_graph.add((vocab_uri, RDF.type, DCAT.Dataset))
        for t in vocab_graph_detail.themes:
            cat_graph.add((vocab_uri, DCAT.theme, t))
        cat_graph.add((cat_uri, DCAT.dataset, vocab_uri))
    cat_file = cat_path / "catalog.ttl"
    with open(cat_file, "wb") as f:
        cat_graph.serialize(f, format="turtle")
