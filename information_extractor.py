# information_extractor.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

try:
    import spacy
except Exception:
    spacy = None

EX = Namespace("http://example.org/ontology#")
OWL_CLASS = URIRef("http://www.w3.org/2002/07/owl#Class")

@dataclass
class ExtractionResult:
    graph: Graph
    jsonld: str

class InformationExtractor:
    """
    Extracts RDF triples (and JSON-LD) from raw story text using spaCy + regex
    with guards so class names don’t become Person instances.

    IMPORTANT CHANGE:
    - If a QuakeEvent has magnitude > 7, we scan the text for explicit mentions
      of the required resources and assert `ex:requiresResource` on that event.
      This lets the Story Rewriter's added sentence *actually* resolve the violation.
    """

    def __init__(self,
                 ontology_path: Optional[str] = "ontology.owl",
                 synthetic_when_no_person: bool = True,
                 synthetic_min_people_per_occupation: int = 2):
        self.ontology_path = ontology_path
        self.synthetic_when_no_person = synthetic_when_no_person
        self.synthetic_min_people_per_occupation = synthetic_min_people_per_occupation

        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

        # Regexes
        self.re_city = re.compile(r"\b(Tokyo|Amsterdam|Utrecht)\b", re.I)
        self.re_quake = re.compile(r"\b(earthquake|quake|seismic)\b", re.I)
        self.re_magnitude = re.compile(r"\b(magnitude|scale)\s*(\d+(?:\.\d+)?)\b", re.I)
        self.re_covid = re.compile(r"\b(covid[- ]?19|coronavirus)\b", re.I)
        self.re_firefighter = re.compile(r"\b(firefighter|fire brigade|fire crew|firefighters)\b", re.I)
        self.re_barista = re.compile(r"\b(barista|coffee[- ]?maker)\b", re.I)
        self.re_doctor = re.compile(r"\b(doctor|physician)\b", re.I)
        self.re_arrogant = re.compile(r"\b(arrogant)\b", re.I)
        self.re_polite = re.compile(r"\b(polite)\b", re.I)

        # Match rewriter’s resource sentence (plurals & variants)
        self.res_regex: Dict[URIRef, re.Pattern] = {
            EX.Firetruck:           re.compile(r"\bfiretruck(s)?\b", re.I),
            EX.Hose:                re.compile(r"\bhose(s)?\b", re.I),
            EX.BreathingApparatus:  re.compile(r"\bbreathing\s*apparatus\b", re.I),
            EX.Ambulance:           re.compile(r"\bambulance(s)?\b", re.I),
            EX.RescueTools:         re.compile(r"\brescue\s*tool(s)?\b", re.I),
            EX.Comms:               re.compile(r"\bcomms?\b|\bcommunications\b", re.I),
        }

        self.RESERVED_CLASS_LABELS = {
            "Barista","Firefighter","Doctor","Hose","Firetruck","BreathingApparatus","Ambulance",
            "RescueTools","Comms","CoffeeMachine","Water","Electricity","Milk","AltMilk","PPE","Ventilator",
            "Arrogant","Polite","Cowardly","Heroic","Crowded","Sparse",
        }
        self._event_counts: Dict[str, int] = {}

        self.base_graph = Graph()
        if ontology_path:
            try:
                self.base_graph.parse(ontology_path)
            except Exception as e:
                print(f"[InformationExtractor] Warning: could not parse ontology at {ontology_path}: {e}")

    def extract(self, text: str, as_jsonld: bool = True) -> ExtractionResult:
        g = Graph()
        g.bind("ex", EX)
        g.bind("rdfs", RDFS)
        self._seed_basics(g)

        # Cities
        for m in self.re_city.finditer(text):
            city = self._cap(m.group(1))
            g.add((getattr(EX, city), RDF.type, EX.City))
            g.add((getattr(EX, city), RDFS.label, Literal(city)))

        # Events + magnitude + location
        quake_events: List[Tuple[URIRef, Optional[float]]] = []
        for sent in self._sentences(text):
            if self.re_quake.search(sent):
                ev = self._new_event(g, "QuakeEvent")
                city_uri = self._city_in_sentence(sent)
                if city_uri: g.add((ev, EX.locatedIn, city_uri))
                mag = self._find_magnitude(sent)
                if mag is not None:
                    g.add((ev, EX.hasMagnitude, Literal(mag, datatype=XSD.decimal)))
                quake_events.append((ev, mag))
        if self.re_covid.search(text):
            ev = self._new_event(g, "CovidEvent")
            city_uri = self._city_in_sentence(text)
            if city_uri: g.add((ev, EX.locatedIn, city_uri))

        # Persons with occupations (conservative)
        if self.re_firefighter.search(text):
            for i in range(self.synthetic_min_people_per_occupation):
                p = self._new_person(g, f"UnnamedFirefighter_{i+1}")
                g.add((p, EX.hasOccupation, EX.Firefighter))
        if self.re_barista.search(text):
            p = self._new_person(g, "PersonX")
            g.add((p, EX.hasOccupation, EX.Barista))
        if self.re_doctor.search(text):
            p = self._new_person(g, "DoctorX")
            g.add((p, EX.hasOccupation, EX.Doctor))

        # Traits
        trait_target = None
        if self.re_arrogant.search(text):
            trait_target = trait_target or self._ensure_person(g, "CustomerY")
            g.add((trait_target, EX.hasTrait, EX.Arrogant))
            g.add((trait_target, RDF.type, EX.Arrogant))
        if self.re_polite.search(text) and trait_target is None:
            p2 = self._ensure_person(g, "CustomerY")
            g.add((p2, EX.hasTrait, EX.Polite))
            g.add((p2, RDF.type, EX.Polite))

        # Covid & symptoms
        if self.re_covid.search(text):
            g.add((EX.Covid19, RDF.type, EX.Disease))
            g.add((EX.Coughing, RDF.type, EX.Symptom))
            g.add((EX.Fever, RDF.type, EX.Symptom))
            g.add((EX.Covid19, EX.hasSymptom, EX.Coughing))
            g.add((EX.Covid19, EX.hasSymptom, EX.Fever))
            px = self._ensure_person(g, "PatientX")
            g.add((px, RDF.type, EX.Disease))
            g.add((px, RDF.type, EX.Covid19))

        # >>> NEW: attach resources for big quakes IF text mentions them <<<
        if quake_events:
            for ev, mag in quake_events:
                if mag is not None and mag > 7.0:
                    for res, pat in self.res_regex.items():
                        if pat.search(text):
                            g.add((ev, EX.requiresResource, res))

        # Ensure at least one person
        if self.synthetic_when_no_person and not any(g.triples((None, RDF.type, EX.Person))):
            self._new_person(g, "UnnamedPerson_1")

        jsonld = g.serialize(format="json-ld", indent=2)
        jsonld = jsonld if isinstance(jsonld, str) else jsonld.decode("utf-8")
        return ExtractionResult(graph=g, jsonld=jsonld)

    # Helpers
    def _seed_basics(self, g: Graph) -> None:
        for cls in [EX.Person, EX.PersonTrait, EX.Occupation, EX.Resource, EX.City, EX.Population,
                    EX.Facility, EX.Protocol, EX.Landmark, EX.Event, EX.QuakeEvent, EX.CovidEvent,
                    EX.Disease, EX.Symptom, EX.Barista, EX.Firefighter, EX.Doctor, EX.Arrogant,
                    EX.Polite, EX.Crowded, EX.Sparse,
                    EX.Firetruck, EX.Hose, EX.BreathingApparatus, EX.Ambulance, EX.RescueTools, EX.Comms]:
            g.add((cls, RDF.type, OWL_CLASS))
        for occ in [EX.Barista, EX.Firefighter, EX.Doctor]:
            g.add((occ, RDFS.subClassOf, EX.Occupation))

    def _sentences(self, text: str) -> List[str]:
        if self.nlp: return [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _cap(self, s: str) -> str:
        return s[:1].upper() + s[1:]

    def _city_in_sentence(self, sent: str) -> Optional[URIRef]:
        m = self.re_city.search(sent)
        return getattr(EX, self._cap(m.group(1))) if m else None

    def _find_magnitude(self, sent: str) -> Optional[float]:
        m = self.re_magnitude.search(sent)
        if m:
            try: return float(m.group(2))
            except Exception: return None
        return None

    def _new_person(self, g: Graph, label: str) -> URIRef:
        if label in self.RESERVED_CLASS_LABELS:
            label = f"Person_{label}"
        uri = getattr(EX, re.sub(r"[^A-Za-z0-9_]", "_", label))
        g.add((uri, RDF.type, EX.Person))
        g.add((uri, RDFS.label, Literal(label)))
        return uri

    def _ensure_person(self, g: Graph, label: str) -> URIRef:
        local = label
        uri = getattr(EX, re.sub(r"[^A-Za-z0-9_]", "_", local))
        if (uri, RDF.type, EX.Person) not in g:
            if local in self.RESERVED_CLASS_LABELS:
                local = f"Person_{local}"
                uri = getattr(EX, re.sub(r"[^A-Za-z0-9_]", "_", local))
            g.add((uri, RDF.type, EX.Person))
            g.add((uri, RDFS.label, Literal(local)))
        return uri

    def _new_event(self, g: Graph, event_type: str) -> URIRef:
        n = self._event_counts.get(event_type, 0) + 1
        self._event_counts[event_type] = n
        local = f"{event_type}_{n}"
        uri = getattr(EX, re.sub(r"[^A-Za-z0-9_]", "_", local))
        g.add((uri, RDF.type, getattr(EX, event_type)))
        return uri
