import unittest

import rdflib
from rdflib.namespace import RDFS, SKOS

from src.harvesters.vocabularies import VocabHarvester


class CanonicalSchemeMembershipTest(unittest.TestCase):
    def test_canonical_scheme_defines_harvested_concept(self) -> None:
        graph = rdflib.Graph()
        concept = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/example")
        canonical_scheme = rdflib.URIRef(
            "http://linked.data.gov.au/def/tern-cv/canonical"
        )
        secondary_scheme = rdflib.URIRef(
            "http://linked.data.gov.au/def/tern-cv/secondary"
        )
        harvester = VocabHarvester.__new__(VocabHarvester)
        harvester.concept_maps = {
            concept: {"concept_schemes": {canonical_scheme, secondary_scheme}}
        }

        harvester.add_canonical_scheme_membership(graph, {concept}, canonical_scheme)

        self.assertIn((concept, SKOS.inScheme, canonical_scheme), graph)
        self.assertIn((concept, RDFS.isDefinedBy, canonical_scheme), graph)
        self.assertNotIn((concept, RDFS.isDefinedBy, secondary_scheme), graph)

    def test_single_scheme_concept_does_not_gain_is_defined_by(self) -> None:
        graph = rdflib.Graph()
        concept = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/example")
        scheme = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/scheme")
        harvester = VocabHarvester.__new__(VocabHarvester)
        harvester.concept_maps = {concept: {"concept_schemes": {scheme}}}

        harvester.add_canonical_scheme_membership(graph, {concept}, scheme)

        self.assertIn((concept, SKOS.inScheme, scheme), graph)
        self.assertNotIn((concept, RDFS.isDefinedBy, scheme), graph)


if __name__ == "__main__":
    unittest.main()
