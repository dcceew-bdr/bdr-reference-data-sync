# TERN reference resources: normalisation currently performed by the BDR

**Date:** 27 August 2026
**Scope:** Pipeline normalisatin of TERN controlled vocabularies from `tern_vocabs_core`; BDR-side normalisation snapshots of TERN Ontology and validators

## Executive summary

The BDR currently transforms harvested TERN vocabulary, ontologuy and shapes data because the source repository and the BDR publication model use different units of organisation. The BDR publishes a vocabulary as a self-contained SKOS Concept Scheme conforming to the VocPub profile. The TERN source graph also uses collections, nested collections, shared concepts and, in some cases, ontology-style packaging. Scheme membership and ownership are not always explicit enough for a consumer to extract one distinct vocabulary document. And TERN Ontologu and validators lack some OntPub and ValPub specified publication-ready metadata.

The normalisation is therefore doing more than RDF syntax cleanup. It currently:

- constructs one output graph per Concept Scheme;
- reconstructs concept, collection and top-concept membership within that graph;
- makes implicit membership explicit so that standard scheme-based queries retrieve complete content;
- selects a canonical scheme in the limited cases where the source graph supplies sufficient evidence;
- corrects several RDF node-kind and property-use problems;
- removes a small amount of redundant or self-referential data; and
- adds publication and processing metadata required by BDR/VocPub operations.

The proposed end state is for source-owned facts to be corrected and maintained by TERN, with each vocabulary available as a profile-conformant artefact at a persistent IRI. The BDR will then be able to retrieve, validate and load an upstream artefact without changing its substantive graph.

Not every current transformation belongs upstream. BDR ingestion provenance, BDR catalogue membership, generated helper schemes, local graph names, themes and keywords are consumer-side concerns. Conversely, vocabulary identity, scheme membership, canonical ownership, source citations, mappings, dates and source-agent metadata should normally be maintained at source.

## What the publication profiles mean here

The profiles are packaging and governance conventions layered on established RDF models; they do not replace SKOS, SHACL or OWL.

- **VocPub** treats one vocabulary document as one `skos:ConceptScheme`, requires the content belonging to that vocabulary to be packaged together, and requires enough metadata for identification, governance, versioning and catalogue use. In particular, VocPub requires one Concept Scheme per vocabulary/file, created and modified dates, creator and publisher IRIs, and consistent structural links. See the [VocPub profile](https://linked.data.gov.au/def/vocpub) and [specification](https://linked.data.gov.au/def/vocpub/spec).
- **ValPub** is a publication profile for SHACL validators. It applies the same general idea to a shapes graph. See the [ValPub repository](https://github.com/AGLDWG/valpub-profile).
- **OntPub** applies the same general idea to a OWL/RDFS ontologies. See the [OntPub profile](https://linked.data.gov.au/def/ontpub) and [specification](https://linked.data.gov.au/def/ontpub/spec).

Some separate validator work has already been undertaken in the BDR catalogue rather than in ingestion pipelines. [bdr-catalogues #16](https://github.com/dcceew-bdr/bdr-catalogues/issues/16) tracked validation and repair of catalogue-held validators against ValPub and SHACL profiling packaging recommendations. Its recorded results include the TERN Ontology Shapes validator, `<https://w3id.org/tern/shapes/tern>`, as ValPub-valid after [a local catalogue correction](https://github.com/dcceew-bdr/bdr-catalogues/commit/1fca7db65a206d1bc80210eb2852e48c85562aab). That work is a useful starting point for an upstream TERN change, but does not by itself establish that TERN's current source artefact contains the same corrections.

## Current harvesting and normalisation

The BDR currently reads the TERN GraphDB SPARQL endpoint, discovers Concept Schemes, Concepts and Collections, determines hierarchy and membership, filters the material for each output vocabulary, obtains descriptions of the selected resources, and serialises a separate graph for each harvested scheme. During that process it performs these changes:

| Area | Current BDR behaviour |
|---|---|
| Vocabulary packaging | Emits one RDF graph/file for each `skos:ConceptScheme`. Where source concepts cannot be assigned to a real scheme, the harvester can create synthetic `urn:vocpub:*` helper schemes. |
| Concept membership | Removes harvested `skos:inScheme` values and reconstructs the value for the output scheme. |
| Collection membership | Reconstructs `skos:member` links for applicable collections, including nested collections, and adds `skos:inScheme` to collections in the output vocabulary. |
| Concepts reachable only through collections | Traverses nested collection structure so that concepts otherwise invisible to `?concept skos:inScheme ?scheme` harvesting are still included. |
| Top concepts | Removes source `skos:hasTopConcept` and `skos:topConceptOf` statements and reconstructs both directions from the filtered hierarchy. |
| Concepts in more than one scheme | Where a TERN concept belongs to multiple schemes and is a top concept of exactly one, the harvester treats that scheme as canonical. For single-scheme output it and adds `rdfs:isDefinedBy` for the canonical scheme. Cases without supporting evidence are not resolved by automation. |
| Bibliographic sources | Changes a non-web, non-IRI `dcterms:source` value on a Concept to `schema:citation`. |
| SKOS mappings | Converts HTTP(S) literals used as objects of `skos:exactMatch`, `relatedMatch` etc. to IRIs. |
| URI-valued literals | For HTTP(S) literals used with `schema:citation`, `schema:url` and `schema:email`, adds the `xsd:anyURI` datatype. |
| Redundant broad relations | Removes explicit `skos:semanticRelation` from copied schemes, collections and concepts before retaining/reconstructing the more specific relationships. It also removes `dcterms:hasPart` from scheme and collection descriptions. |
| Self-equivalence | Removes `owl:sameAs` where subject and object are identical. |
| Source agents | Adds `schema:creator <https://linked.data.gov.au/org/tern>` and `schema:publisher <https://linked.data.gov.au/org/dcceew>` to schemes. |
| Namespace metadata | Adds `vann:preferredNamespacePrefix` and `vann:preferredNamespaceUri`. |
| BDR processing provenance | Adds a processing note and a PROV activity identifying the BDR ingestion script and timestamp. |
| BDR catalogue enrichment | Adds local catalogue relationships, graph names, themes, keywords and serialisation tokens. |

## Priority upstream changes to offer TERN

### 1. Establish one authoritative publication artefact per vocabulary

Each vocabulary should be downloadable as a self-contained RDF artefact centred on exactly one `skos:ConceptScheme`. The artefact should contain no unrelated vocabulary descriptions.

At minimum the scheme should have a persistent IRI, type, preferred label, definition, created date, maintained modified date, creator and publisher IRIs, and required top-concept structure. A source-controlled validator should be run against the exact artefact.

This important structural change turns synchronisation into retrieval of an authoritative resource rather than reconstruction from a shared triple store.

### 2. Make scheme membership complete and explicit

The most consequential present gap is content that is discoverable through collection traversal but not through ordinary scheme membership. The source graph should explicitly link to its scheme:

```turtle
<concept> a skos:Concept ;
    skos:inScheme <scheme> .

<collection> a skos:Collection ;
    skos:inScheme <scheme> .
```

This applies to nested Collections as well as leaf Collections and their Concept members. Existing `skos:member` structure should remain; the proposed triples make ownership/discoverability explicit and do _not_ flatten that structure.

The June 2026 audit recorded in [bdr-ops #243](https://github.com/dcceew-bdr/bdr-ops/issues/243) found 157 TERN Collections without `skos:inScheme`, 152 collection-to-collection links, and 1,317 Concepts reachable through collection membership with no `skos:inScheme`. That audit should be rerun against the artefacts used prior to any pull requests.

### 3. Resolve canonical ownership of shared concepts

SKOS permits a Concept to be associated with more than one Concept Scheme, but VocPub publication is deliberately document/scheme-centred. TERN Concept IRI structure does not encode their scheme, so consumers need explicit graph statements to determine Concept definition and scheme membership.

The same June audit found 60 Concepts with two scheme memberships. For 13, being a top concept in only one scheme gave plausible evidence of canonical ownership; 47 still require curatorial decisions, including one Concept that was a top concept of both schemes. BDR normalisation resolves only the unique-evidence cases. TERN should confirm those 13 and decide the remainder rather than having the BDR institutionalise guesses.

A useful pattern for an unchanged Concept that is defined by one scheme but intentionally reused in another is:

```turtle
<concept> a skos:Concept ;
    skos:inScheme <canonical-scheme>, <reusing-scheme> ;
    rdfs:isDefinedBy <canonical-scheme> .
```

The reused Concept must also appear in the reusing vocabulary's hierarchy as required by VocPub. Depending on the publication workflow, its local `skos:inScheme` can be authored explicitly or supplied by the VocPub expansion rules from that hierarchy. `rdfs:isDefinedBy` disambiguates the defining scheme; it does _not_ prevent membership in another scheme.

`prov:wasDerivedFrom` should be added where there is actual derivation. If the second vocabulary creates a distinct, adapted Concept with its own IRI, the relationship should be expressed from the new Concept to the source Concept:

```turtle
<adapted-concept> a skos:Concept ;
    skos:inScheme <adapting-scheme> ;
    rdfs:isDefinedBy <adapting-scheme> ;
    prov:wasDerivedFrom <source-concept> .
```

Using `prov:wasDerivedFrom` from a reused Concept to itself would be incorrect: the same IRI denotes the same Concept, not a newly derived entity. Where an entire vocabulary is derived from another vocabulary, VocPub requirement 2.1.11 recommends `prov:qualifiedDerivation` on the Concept Scheme, with `prov:entity` identifying the source vocabulary and `prov:hadRole` identifying a mode from the Vocabulary Derivation Modes vocabulary—for example, `https://linked.data.gov.au/def/vocdermods/extension`. A simple `prov:wasDerivedFrom` on the scheme is also valid origin metadata under requirement 2.1.07.

The modelling policy should therefore cover three separate cases: multi-scheme membership of one unchanged Concept; reuse of that Concept in another vocabulary document; and creation of a distinct derived Concept or vocabulary.

### 4. Correct value types and property use

These changes are mostly local and low risk:

- replace bibliographic-text `dcterms:source` values with `schema:citation`;
- express the targets of SKOS mapping properties as IRIs, not string or `xsd:anyURI` literals;
- type URI-valued literals as required for `schema:citation`, `schema:url` and `schema:email`; and
- remove self-referential `owl:sameAs` statements.

The mapping review has a semantic component: converting a literal to an IRI repairs its RDF node kind, but does not prove that `skos:exactMatch` or another mapping property is the right relationship. Targets that are documents rather than concepts should be reviewed separately.

### 5. Complete and maintain source metadata

For each scheme, TERN and DCCEEW should agree the creator, publisher and any custodian/attribution roles, using resolvable agent IRIs. Source authorship must remain distinct from BDR ingestion provenance and BDR republication.

Created dates should remain stable. Modified dates (and any explicit version identifiers) should change when the published graph changes and should be machine-readable, so that the proposed synchroniser can reload only when the version or modified date changes.

The following are not all current automated fixes and should be treated as review items:

- prefer explicit created/modified properties over ambiguous `dcterms:date`;
- audit notation uniqueness and use typed notation systems where necessary;
- confirm whether materialised `skos:semanticRelation`, `mappingRelation` and transitive hierarchy statements are needed by TERN applications before removing them; and
- leave valid exact/close duplicates alone unless TERN wishes to simplify them.

See the original, now-superseded audit discussion in [bdr-vocabs #2](https://github.com/dcceew-bdr/bdr-vocabs/issues/2).

### 6. Publish and resolve the resource IRIs

The vocabulary IRI should resolve to a representation of that vocabulary, with normal HTTP content negotiation or another agreed stable download mechanism. The same principle should apply to the TERN Ontology and each validator. The artefact returned at the persistent IRI should be the one validated and versioned by its source repository.

This is not a graph normalisation rule, but it is what allows the normalisation scripts to be retired. A repository URL and a resource IRI have different lifecycles and roles; BDR synchronisation should depend on the resource IRI.

## What should remain BDR-side

The upstream pull requests should not copy the complete current output of the harvester. The following are derived or deployment-specific:

- BDR ingestion PROV activities and generation timestamps;
- BDR catalogue membership and BDR real/virtual graph identifiers;
- BDR-only themes, keywords, filename tokens and namespace registry entries;
- synthetic `urn:vocpub:in-collections` and `urn:vocpub:concepts` schemes, if neeed;
- filtering needed to exclude resources that BDR publishes through another catalogue; and
- defensive cleanup of extra triples returned by a GraphDB `DESCRIBE` implementation.

The BDR should validate the artefacts and preserve their source graph. BDR catalogue metadata can then be added in a separate named graph or catalogue resource without rewriting the vocabulary itself.

## Evidence and implementation traceability

For current vocbularu normalisation, the umbrella item is [bdr-ops #198](https://github.com/dcceew-bdr/bdr-ops/issues/198). Its sub-issues and the corresponding implementation evidence are below.

| Topic | Issue | BDR implementation evidence | Status/qualification |
|---|---|---|---|
| Complete Concept and Collection membership | [#195](https://github.com/dcceew-bdr/bdr-ops/issues/195), [#196](https://github.com/dcceew-bdr/bdr-ops/issues/196), [#209](https://github.com/dcceew-bdr/bdr-ops/issues/209) | [Add `inScheme` to Collections](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/b55ca4755f6f76ba905e1574a5b8d894188917c2) and existing nested-collection traversal | #195 is the upstream proposal; #196 remains open as the general defensive requirement. |
| Bibliographic source to citation | [#199](https://github.com/dcceew-bdr/bdr-ops/issues/199) | [Concept metadata and mapping cleanup](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/8bb3fea1db0dc56614d08d01411cb96d21bd8c21) | Implemented for non-web, non-IRI `dcterms:source` values on Concepts. |
| Generic HTTP literal typing proposal | [#200](https://github.com/dcceew-bdr/bdr-ops/issues/200) | Superseded in practice by the predicate-specific [#203 implementation](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/807e94b76d0f69e5fecf1d5073ddc5bb672ca72d) | The implementation deliberately does not type every HTTP-looking literal. |
| Mapping literals to IRIs | [#202](https://github.com/dcceew-bdr/bdr-ops/issues/202) | [Concept metadata and mapping cleanup](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/8bb3fea1db0dc56614d08d01411cb96d21bd8c21) | Node-kind correction implemented; semantic appropriateness still needs review. |
| Required `xsd:anyURI` literals | [#203](https://github.com/dcceew-bdr/bdr-ops/issues/203) | [Normalise URI literals for VocPub](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/807e94b76d0f69e5fecf1d5073ddc5bb672ca72d) | Implemented for citation, URL and email. |
| Self-referential `owl:sameAs` | [#206](https://github.com/dcceew-bdr/bdr-ops/issues/206) | [Remove self-referential `owl:sameAs`](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/5e9ebd89dc91d414ecf50154811557ba9a57cbb9) | Implemented. |
| Creator and publisher | [#207](https://github.com/dcceew-bdr/bdr-ops/issues/207) | [Add agents to TERN Concept Schemes](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/bc849e13064d49ce39177a3b868e6a6ffbe4eada) | Implemented as a BDR rule; roles should be confirmed before upstreaming. |
| Preserve canonical definition when removing secondary membership | [#208](https://github.com/dcceew-bdr/bdr-ops/issues/208) | [Initial `rdfs:isDefinedBy` handling](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/606e19013e19c5ca30b763aaa2cfa0c39ea5c2ad), [canonical attribution correction](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/17aeda30342c5eaf04dc8572bd96b648b474e1ee) | Implemented for harvested output. |
| Canonical scheme ambiguity | [#243](https://github.com/dcceew-bdr/bdr-ops/issues/243) | [Prefer a unique top-concept scheme](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/b68b2cef7a933703a0897af95af98e692e565f37), [post-harvest enforcement and diagnostics](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/6ca7b5f2dba28c13912b7a4d7717f7f44153e6b5) | Automation covers only cases with unique structural evidence; curatorial cases remain. |
| BDR transformation provenance | Not a child issue | [Add ingestion provenance metadata](https://github.com/dcceew-bdr/bdr-reference-data-sync/commit/2c3440df388011120709e06f0d21e99728afc993) | BDR-side only, not TERN source provenance. |

These issues and commits are the authoritative implementation trail; future TERN work can create reviewable upstream pull-request trails.

## Proposed delivery sequence

1. Agree the ownership and packaging rules: one artefact per scheme, treatment of shared Concepts and Collections, agent roles, and date/version policy.
2. Re-run diagnostics against TERN’s current source artefacts - do not treat the June audit as current without verification.
3. Prepare small, reviewable TERN pull requests in this order: 
    a. metadata/value-kind corrections; 
    b. explicit collection and concept membership; 
    c. canonical ownership decisions.
4. Add or confirm the current VocPub validator to TERN CI and validate the exact distributable files. Where a requirement is genuinely unsuitable, raise in [VocPub issues](https://github.com/AGLDWG/vocpub-profile/issues).
5. In parallel, assess the TERN Ontology against OntPub and the RLP validators against ValPub using separate change sets. For the TERN Ontology Shapes validator, begin with the ValPub work recorded in [bdr-catalogues #16](https://github.com/dcceew-bdr/bdr-catalogues/issues/16), compare that updated catalogue copy with TERN's current source, and upstream applicable changes.
6. Configure persistent IRI resolution to serve the validated artefacts, including clear modified/version signals.
7. Change the BDR sync to poll and validate those IRIs, compare version/modified metadata, and load unchanged source graphs. Retain only BDR catalogue metadata outside those graphs.

## Requested Decisions

- Does TERN agree to one maintained, directly downloadable RDF artefact per Concept Scheme?
- Which scheme canonically defines each currently shared Concept, and which additional scheme memberships are intentional?
- Which scheme owns each Collection that is currently unscoped or reused?
- Who should be recorded as creator, publisher and custodian for each family of resources?
- What event updates `schema:dateModified`/`dcterms:modified`, and should a separate version IRI or version string also maintained?
- Can the persistent resource IRIs resolve directly to the validated RDF artefacts?
- Are any current TERN applications dependent on materialised generic/transitive SKOS relationships that the BDR presently removes?
- Are any VocPub, ValPub or OntPub requirements problematic for TERN’s authoring and release workflows? If so, can those cases be brought back as explicit profile change proposals?