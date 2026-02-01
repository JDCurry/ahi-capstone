"""
HAZARD-LM v2.0: Data Processing Utilities

Handles:
- Loading and preprocessing hazard data from various sources
- FEMA, NWS, NOAA data formats
- Compound event extraction
- Train/val/test splitting with stratification
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import re
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


# =============================================================================
# Data Schemas
# =============================================================================

@dataclass
class HazardSample:
    """Single hazard data sample."""
    text: str
    hazards: List[str]  # primary hazard(s)
    severity: Optional[int] = None  # 1-5 scale
    compound_event: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'hazards': self.hazards,
            'severity': self.severity,
            'compound_event': self.compound_event,
            'metadata': self.metadata
        }


# =============================================================================
# Hazard Taxonomy
# =============================================================================

class HazardTaxonomy:
    """Defines hazard types, subtypes, and relationships."""
    
    PRIMARY_HAZARDS = {
        'fire': ['wildfire', 'structure_fire', 'wui_fire'],
        'flood': ['riverine', 'flash', 'coastal', 'dam_failure'],
        'earthquake': ['tectonic', 'induced', 'aftershock'],
        'wind': ['tornado', 'hurricane', 'derecho', 'straight_line'],
        'freeze': ['ice_storm', 'frost', 'cold_snap'],
        'heat': ['heat_wave', 'excessive_heat'],
        'drought': ['agricultural', 'hydrological', 'meteorological'],
    }
    
    # Compound events: hazard combinations that interact
    COMPOUND_EVENTS = {
        'wind_fire': {'components': ('wind', 'fire'), 'description': 'Wind-driven wildfire'},
        'earthquake_flood': {'components': ('earthquake', 'flood'), 'description': 'Quake-triggered flood (dam failure)'},
        'drought_fire': {'components': ('drought', 'fire'), 'description': 'Drought-preconditioned fire'},
        'freeze_flood': {'components': ('freeze', 'flood'), 'description': 'Ice dam breakup flood'},
        'heat_fire': {'components': ('heat', 'fire'), 'description': 'Heat wave fire conditions'},
    }
    
    # Keywords for hazard detection
    KEYWORDS = {
        'fire': ['fire', 'burn', 'wildfire', 'blaze', 'ignition', 'smoke', 'firefighter', 
                 'evacuation', 'containment', 'acres burned', 'red flag warning'],
        'flood': ['flood', 'flooding', 'inundation', 'flash flood', 'river stage', 
                  'levee', 'dam', 'water level', 'storm surge', 'high water'],
        'earthquake': ['earthquake', 'quake', 'seismic', 'tremor', 'magnitude', 
                       'richter', 'epicenter', 'aftershock', 'fault', 'tremblor'],
        'wind': ['wind', 'gust', 'tornado', 'hurricane', 'typhoon', 'derecho', 
                 'wind advisory', 'high wind', 'tropical storm', 'cyclone'],
        'freeze': ['freeze', 'frost', 'ice', 'freezing', 'cold', 'winter storm',
                   'ice storm', 'black ice', 'hypothermia', 'frostbite'],
        'heat': ['heat', 'hot', 'heat wave', 'excessive heat', 'heat advisory',
                 'heat stroke', 'temperature record', 'heat index'],
        'drought': ['drought', 'dry', 'water shortage', 'precipitation deficit',
                    'water restrictions', 'reservoir', 'groundwater'],
    }
    
    @classmethod
    def detect_hazards(cls, text: str) -> List[str]:
        """Detect hazards mentioned in text based on keywords."""
        text_lower = text.lower()
        detected = []
        
        for hazard, keywords in cls.KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if hazard not in detected:
                        detected.append(hazard)
                    break
        
        return detected
    
    @classmethod
    def detect_compound_event(cls, hazards: List[str]) -> Optional[str]:
        """Detect if hazard combination forms a compound event."""
        hazard_set = set(hazards)
        
        for event_name, event_info in cls.COMPOUND_EVENTS.items():
            if set(event_info['components']).issubset(hazard_set):
                return event_name
        
        return None


# =============================================================================
# Data Loaders for Various Sources
# =============================================================================

class NWSDataLoader:
    """Load and parse NWS (National Weather Service) data."""
    
    # NWS product codes to hazard mapping
    PRODUCT_MAPPING = {
        'RFW': 'fire',      # Red Flag Warning
        'FWW': 'fire',      # Fire Weather Watch
        'FFW': 'flood',     # Flash Flood Warning
        'FLW': 'flood',     # Flood Warning
        'SVR': 'wind',      # Severe Thunderstorm Warning
        'TOR': 'wind',      # Tornado Warning
        'HUR': 'wind',      # Hurricane Warning
        'WSW': 'freeze',    # Winter Storm Warning
        'BZW': 'freeze',    # Blizzard Warning
        'EHW': 'heat',      # Excessive Heat Warning
        'HEA': 'heat',      # Heat Advisory
    }
    
    @classmethod
    def parse_nws_product(cls, text: str, product_code: str = None) -> HazardSample:
        """Parse NWS product text into HazardSample."""
        # Detect hazards from product code or text
        hazards = []
        
        if product_code and product_code in cls.PRODUCT_MAPPING:
            hazards.append(cls.PRODUCT_MAPPING[product_code])
        
        # Also detect from text
        text_hazards = HazardTaxonomy.detect_hazards(text)
        for h in text_hazards:
            if h not in hazards:
                hazards.append(h)
        
        # Detect compound events
        compound = HazardTaxonomy.detect_compound_event(hazards)
        
        return HazardSample(
            text=text,
            hazards=hazards,
            compound_event=compound,
            metadata={'source': 'NWS', 'product_code': product_code}
        )


class FEMADataLoader:
    """Load and parse FEMA incident data."""
    
    # FEMA incident type mapping
    INCIDENT_MAPPING = {
        'Fire': 'fire',
        'Flood': 'flood',
        'Earthquake': 'earthquake',
        'Severe Storm': 'wind',
        'Tornado': 'wind',
        'Hurricane': 'wind',
        'Winter Storm': 'freeze',
        'Severe Ice Storm': 'freeze',
        'Drought': 'drought',
    }
    
    @classmethod
    def parse_fema_incident(cls, incident: Dict) -> HazardSample:
        """Parse FEMA incident record into HazardSample."""
        incident_type = incident.get('incidentType', '')
        description = incident.get('declarationTitle', '')
        
        # Map incident type to hazard
        hazards = []
        for fema_type, hazard in cls.INCIDENT_MAPPING.items():
            if fema_type.lower() in incident_type.lower():
                hazards.append(hazard)
        
        # Also check description
        text_hazards = HazardTaxonomy.detect_hazards(description)
        for h in text_hazards:
            if h not in hazards:
                hazards.append(h)
        
        return HazardSample(
            text=description,
            hazards=hazards or ['unknown'],
            compound_event=HazardTaxonomy.detect_compound_event(hazards),
            metadata={
                'source': 'FEMA',
                'incident_type': incident_type,
                'state': incident.get('state'),
                'declaration_date': incident.get('declarationDate'),
            }
        )


# =============================================================================
# PyTorch Dataset
# =============================================================================

class HazardDataset(Dataset):
    """PyTorch Dataset for hazard text classification."""
    
    def __init__(
        self,
        samples: List[HazardSample],
        tokenizer: Any,  # HuggingFace tokenizer
        max_length: int = 512,
        hazards: List[str] = None,
        include_compound: bool = True
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hazards = hazards or ['fire', 'flood', 'earthquake', 'wind', 'freeze']
        self.include_compound = include_compound
        
        # Create label mappings
        self.hazard_to_idx = {h: i for i, h in enumerate(self.hazards)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            sample.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # Create labels for each hazard (multi-label)
        for hazard in self.hazards:
            label = 1 if hazard in sample.hazards else 0
            item[f'{hazard}_labels'] = torch.tensor(label, dtype=torch.long)
        
        # Compound event labels
        if self.include_compound and sample.compound_event:
            item[f'{sample.compound_event}_labels'] = torch.tensor(1, dtype=torch.long)
        
        return item


class SimpleTokenizer:
    """Simple tokenizer for testing (no HuggingFace dependency)."""
    
    def __init__(self, vocab_size: int = 50000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_id = {}
        self.next_id = 1  # 0 reserved for padding
    
    def __call__(
        self,
        text: str,
        max_length: int = None,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        max_len = max_length or self.max_length
        # Support both single string and list/tuple of strings (batch)
        if isinstance(text, (list, tuple)):
            texts = text
        else:
            texts = [text]

        all_ids = []
        all_masks = []
        for t in texts:
            # Simple tokenization: split on whitespace
            words = t.lower().split()

            # Convert to IDs
            ids = []
            for word in words[:max_len]:
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id % self.vocab_size
                    self.next_id += 1
                ids.append(self.word_to_id[word])

            # Pad or truncate
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            else:
                ids = ids[:max_len]

            attention_mask = [1 if i != 0 else 0 for i in ids]

            all_ids.append(ids)
            all_masks.append(attention_mask)

        return {
            'input_ids': torch.tensor(all_ids),
            'attention_mask': torch.tensor(all_masks)
        }


# =============================================================================
# Data Splitting Utilities
# =============================================================================

def stratified_split(
    samples: List[HazardSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[HazardSample], List[HazardSample], List[HazardSample]]:
    """
    Split samples with stratification by hazard type.
    
    Ensures each split has similar hazard distribution.
    """
    random.seed(seed)
    
    # Group by primary hazard
    hazard_groups = defaultdict(list)
    for sample in samples:
        primary = sample.hazards[0] if sample.hazards else 'unknown'
        hazard_groups[primary].append(sample)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for hazard, group_samples in hazard_groups.items():
        random.shuffle(group_samples)
        n = len(group_samples)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples.extend(group_samples[:train_end])
        val_samples.extend(group_samples[train_end:val_end])
        test_samples.extend(group_samples[val_end:])
    
    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


def per_hazard_split(
    samples: List[HazardSample],
    hazards: List[str],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, Dict[str, List[HazardSample]]]:
    """
    Create per-hazard datasets for independent adapter training.
    
    Returns:
        Dict mapping hazard name to {'train': [...], 'val': [...]}
    """
    random.seed(seed)
    
    result = {}
    
    for hazard in hazards:
        # Filter samples containing this hazard
        hazard_samples = [s for s in samples if hazard in s.hazards]
        random.shuffle(hazard_samples)
        
        split_idx = int(len(hazard_samples) * train_ratio)
        
        result[hazard] = {
            'train': hazard_samples[:split_idx],
            'val': hazard_samples[split_idx:]
        }
    
    return result


def extract_compound_events(
    samples: List[HazardSample]
) -> List[HazardSample]:
    """Extract samples with compound events for interaction training."""
    return [s for s in samples if s.compound_event is not None]


# =============================================================================
# Sample Data Generator (for testing)
# =============================================================================

def generate_sample_data(
    n_samples: int = 1000,
    hazards: List[str] = None
) -> List[HazardSample]:
    """Generate synthetic hazard data for testing."""
    hazards = hazards or ['fire', 'flood', 'earthquake', 'wind', 'freeze']
    
    # Sample templates
    templates = {
        'fire': [
            "Wildfire burning {acres} acres in {location}. {containment}% contained.",
            "Red flag warning issued due to high winds and low humidity.",
            "Fire weather conditions expected with gusts up to {wind} mph.",
            "Evacuation orders issued for {location} due to approaching wildfire.",
        ],
        'flood': [
            "Flash flood warning for {location}. {inches} inches of rain expected.",
            "River stages rising at {location}. Flood stage expected by {time}.",
            "Coastal flooding expected due to storm surge from {storm}.",
            "Dam levels at {percent}% capacity. Controlled release planned.",
        ],
        'earthquake': [
            "Magnitude {mag} earthquake reported {distance} miles from {location}.",
            "Seismic activity detected near {fault} fault line.",
            "Aftershock sequence continuing following M{mag} mainshock.",
            "Earthquake early warning issued for {region}.",
        ],
        'wind': [
            "Severe thunderstorm warning with {wind} mph wind gusts.",
            "Tornado watch issued for {location}. Conditions favorable.",
            "Hurricane {name} approaching with sustained winds of {wind} mph.",
            "High wind warning: gusts up to {wind} mph expected.",
        ],
        'freeze': [
            "Winter storm warning: {inches} inches of snow expected.",
            "Ice storm warning for {location}. Significant ice accumulation.",
            "Freeze warning: temperatures dropping to {temp}°F overnight.",
            "Blizzard conditions expected with visibility near zero.",
        ],
    }
    
    samples = []
    
    for _ in range(n_samples):
        # Random primary hazard
        primary_hazard = random.choice(hazards)
        template = random.choice(templates[primary_hazard])
        
        # Fill template
        text = template.format(
            acres=random.randint(100, 50000),
            location=random.choice(['Northern California', 'Texas', 'Florida', 'Montana', 'Oregon']),
            containment=random.randint(0, 100),
            wind=random.randint(30, 150),
            inches=random.uniform(1, 10),
            time=random.choice(['noon', 'evening', 'overnight', 'tomorrow']),
            storm=random.choice(['tropical storm', 'hurricane', 'nor\'easter']),
            percent=random.randint(50, 100),
            mag=round(random.uniform(3.0, 8.0), 1),
            distance=random.randint(5, 100),
            fault=random.choice(['San Andreas', 'Hayward', 'New Madrid']),
            region=random.choice(['Bay Area', 'Los Angeles', 'Pacific Northwest']),
            name=random.choice(['Ian', 'Maria', 'Harvey', 'Sandy']),
            temp=random.randint(-20, 32),
        )
        
        # Maybe add secondary hazard (20% chance)
        hazard_list = [primary_hazard]
        if random.random() < 0.2:
            secondary = random.choice([h for h in hazards if h != primary_hazard])
            hazard_list.append(secondary)
        
        sample = HazardSample(
            text=text,
            hazards=hazard_list,
            severity=random.randint(1, 5),
            compound_event=HazardTaxonomy.detect_compound_event(hazard_list),
            metadata={'source': 'synthetic'}
        )
        
        samples.append(sample)
    
    return samples


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Generate sample data
    samples = generate_sample_data(n_samples=1000)
    
    print(f"Generated {len(samples)} samples")
    
    # Count hazards
    hazard_counts = defaultdict(int)
    for sample in samples:
        for hazard in sample.hazards:
            hazard_counts[hazard] += 1
    
    print("\nHazard distribution:")
    for hazard, count in sorted(hazard_counts.items()):
        print(f"  {hazard}: {count}")
    
    # Count compound events
    compound_counts = defaultdict(int)
    for sample in samples:
        if sample.compound_event:
            compound_counts[sample.compound_event] += 1
    
    print("\nCompound events:")
    for event, count in sorted(compound_counts.items()):
        print(f"  {event}: {count}")
    
    # Test dataset creation
    tokenizer = SimpleTokenizer()
    dataset = HazardDataset(samples[:100], tokenizer)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample item keys: {list(dataset[0].keys())}")
