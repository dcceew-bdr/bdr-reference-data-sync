import unittest
from unittest.mock import Mock, patch

import rdflib

from src.impl.prez_manifest import load_remote_rdf_artifact


class RemotePrezManifestArtifactTest(unittest.TestCase):
    @patch("src.impl.prez_manifest.httpx.Client")
    def test_loads_https_turtle_artifact(self, client_class: Mock) -> None:
        response = Mock()
        response.content = b"<https://example.com/s> <https://example.com/p> <https://example.com/o> ."
        response.raise_for_status.return_value = None
        client_class.return_value.__enter__.return_value.get.return_value = response
        graph = rdflib.Graph()

        load_remote_rdf_artifact("https://data.example/asgs.ttl", graph)

        self.assertEqual(len(graph), 1)
        client_class.return_value.__enter__.return_value.get.assert_called_once_with(
            "https://data.example/asgs.ttl",
            headers={"Accept": "text/turtle, application/rdf+xml, application/ld+json"},
        )

    def test_rejects_remote_nquads_for_single_graph_build(self) -> None:
        with self.assertRaises(NotImplementedError):
            load_remote_rdf_artifact("https://data.example/asgs.nq", rdflib.Graph())


if __name__ == "__main__":
    unittest.main()
