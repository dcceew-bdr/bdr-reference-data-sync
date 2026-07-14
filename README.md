# BDR Reference Data Sync repo

This repo contains the code for the BDR Reference Data Sync scripts.

This set of scripts harvests and catalogs Controlled Vocabularies, Instance Lists, Codelists, Ontologies and other reference data for the BDR from a variety of sources.

## BDR-owned catalogue content

BDR-owned vocabularies and small static reference datasets are maintained in
the `bdr-catalogues` source repository and loaded through its Prez Manifest.
They are not harvested a second time by this project. Remote third-party
vocabularies continue to be harvested from their configured upstream sources.

An individual catalogue can be built while validating a source change:

```shell
.venv/bin/python main.py build --catalogue bdr-cat
```

## Maintenance commands

Audit explicit `owl:sameAs` triples where the subject and object are identical in configured GraphDB sources:

```shell
python main.py fix-self-sameas
```

Apply the upstream delete through the configured GraphDB update endpoint:

```shell
python main.py fix-self-sameas --catalogue tern-cv --apply
```
