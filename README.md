# BDR Reference Data Sync repo

This repo contains the code for the BDR Reference Data Sync scripts.

This set of scripts harvests and catalogs Controlled Vocabularies, Instance Lists, Codelists, Ontologies and other reference data for the BDR from a variety of sources.

## BDR-owned catalogue content

BDR-owned vocabularies and small static reference datasets are maintained in
the `bdr-catalogues` source repository and loaded through its Prez Manifest.
They are not harvested a second time by this project. Remote third-party
vocabularies continue to be harvested from their configured upstream sources.

Prez Manifest artifacts may be local RDF paths or HTTP(S) URLs. This allows
large generated datasets to remain in controlled object storage while their
catalogue metadata and artifact declaration remain versioned with the BDR
catalogues. Remote artifacts must have a recognised RDF filename extension;
the build follows redirects and fails if the download or RDF parse fails.

An individual catalogue can be built while validating a source change:

```shell
.venv/bin/python main.py build --catalogue bdr-cat
```

Validate the generated TERN graph for concepts with multiple or missing
`skos:inScheme` values and collections with missing `skos:inScheme` values:

```shell
./scripts/validate-tern-scheme-membership.sh generated/tern-cv_all.nq
```

Local validation exits with status 1 when findings are present, so it can be
used as a post-build gate. The report remains useful while known upstream TERN
membership issues are being curated.

Run the same validation after deployment by supplying the repository SPARQL
endpoint instead of the generated N-Quads path.

## Maintenance commands

Audit explicit `owl:sameAs` triples where the subject and object are identical in configured GraphDB sources:

```shell
python main.py fix-self-sameas
```

Apply the upstream delete through the configured GraphDB update endpoint:

```shell
python main.py fix-self-sameas --catalogue tern-cv --apply
```
