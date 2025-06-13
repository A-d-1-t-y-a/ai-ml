"""
Simple Demo for Semester NER System
Works with basic dependencies for testing core functionality
"""

import os
import sys
import json
import re
from typing import List, Dict

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {title.center(58)} ")
    print("="*60)

def print_step(step, description):
    """Print step information"""
    print(f"\n--- Step {step}: {description} ---")

class SimpleTextProcessor:
    """Simple text processor without external dependencies"""
    
    def __init__(self):
        # Semester patterns for recognition
        self.semester_patterns = {
            'numeric_semester': r'\b(?:semester|sem)\s*[1-8](?:st|nd|rd|th)?\b',
            'ordinal_semester': r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s*semester\b',
            'season_semester': r'\b(?:fall|spring|summer|winter)\s*(?:semester|term)\b',
        }
    
    def extract_semester_entities(self, text: str) -> List[Dict]:
        """Extract semester entities using pattern matching"""
        entities = []
        text_lower = text.lower()
        
        for pattern_name, pattern in self.semester_patterns.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            
            for match in matches:
                entity = {
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern_type': pattern_name,
                    'label': self._classify_semester_entity(match.group())
                }
                entities.append(entity)
        
        return entities
    
    def _classify_semester_entity(self, entity_text: str) -> str:
        """Classify semester entity"""
        entity_lower = entity_text.lower()
        
        # Check for specific semester numbers
        for i in range(1, 9):
            if f'semester {i}' in entity_lower or f'sem {i}' in entity_lower:
                return f'SEMESTER_{i}'
        
        # Check for season-based semesters
        if 'fall' in entity_lower:
            return 'FALL_SEMESTER'
        elif 'spring' in entity_lower:
            return 'SPRING_SEMESTER'
        elif 'summer' in entity_lower:
            return 'SUMMER_SEMESTER'
        
        # Check for ordinal semesters
        ordinal_map = {
            'first': 'SEMESTER_1', 'second': 'SEMESTER_2', 'third': 'SEMESTER_3',
            'fourth': 'SEMESTER_4', 'fifth': 'SEMESTER_5', 'sixth': 'SEMESTER_6',
            'seventh': 'SEMESTER_7', 'eighth': 'SEMESTER_8'
        }
        
        for ordinal, label in ordinal_map.items():
            if ordinal in entity_lower:
                return label
        
        return 'OTHER'

class SimpleSemesterNER:
    """Simple NER system for demonstration"""
    
    def __init__(self):
        self.processor = SimpleTextProcessor()
        self.stats = {'total_processed': 0, 'entities_found': 0}
    
    def process_text(self, text: str) -> Dict:
        """Process text and extract semester entities"""
        self.stats['total_processed'] += 1
        
        # Extract entities
        entities = self.processor.extract_semester_entities(text)
        self.stats['entities_found'] += len(entities)
        
        # Create result
        result = {
            'text': text,
            'entities': entities,
            'entity_count': len(entities),
            'has_semester_info': len(entities) > 0
        }
        
        return result
    
    def batch_process(self, texts: List[str]) -> List[Dict]:
        """Process multiple texts"""
        results = []
        for text in texts:
            result = self.process_text(text)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()

def demo_pattern_matching():
    """Demonstrate pattern-based entity extraction"""
    print_header("PATTERN-BASED ENTITY EXTRACTION DEMO")
    
    ner_system = SimpleSemesterNER()
    
    test_texts = [
        "This course is offered in the first semester of the program.",
        "Students must complete spring semester requirements.",
        "The fall semester includes advanced mathematics courses.",
        "Second semester curriculum focuses on programming.",
        "Summer semester offers intensive workshops.",
        "Students in semester 3 study data structures.",
        "The winter term has special elective courses.",
        "Fourth semester includes project work.",
        "This is a general education requirement.",
        "Academic year consists of multiple semesters."
    ]
    
    print_step(1, "Processing Sample Texts")
    
    results = ner_system.batch_process(test_texts)
    
    # Display results
    print(f"\nProcessed {len(test_texts)} texts:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Text: \"{result['text']}\"")
        
        if result['entities']:
            print(f"   ✅ Found {len(result['entities'])} semester entities:")
            for entity in result['entities']:
                print(f"      → '{entity['text']}' → {entity['label']} ({entity['pattern_type']})")
        else:
            print("   ❌ No semester entities found")
    
    # Show statistics
    stats = ner_system.get_statistics()
    print(f"\n--- Statistics ---")
    print(f"Total texts processed: {stats['total_processed']}")
    print(f"Total entities found: {stats['entities_found']}")
    print(f"Average entities per text: {stats['entities_found']/stats['total_processed']:.2f}")

def demo_entity_classification():
    """Demonstrate entity classification"""
    print_header("ENTITY CLASSIFICATION DEMO")
    
    ner_system = SimpleSemesterNER()
    
    classification_tests = [
        "Students in semester 1 learn programming basics.",
        "The first semester covers foundational topics.",
        "Fall semester 2024 registration is open.",
        "Spring semester includes advanced courses.",
        "Summer semester has intensive programs.",
        "Third semester focuses on algorithms.",
        "The winter term offers special workshops.",
        "Final semester projects are due soon.",
        "This course is a general requirement.",
        "Academic calendar shows all semester dates."
    ]
    
    print_step(1, "Testing Entity Classification")
    
    entity_counts = {}
    
    for text in classification_tests:
        result = ner_system.process_text(text)
        
        print(f"\nText: \"{text}\"")
        
        if result['entities']:
            for entity in result['entities']:
                label = entity['label']
                entity_counts[label] = entity_counts.get(label, 0) + 1
                print(f"  → Entity: '{entity['text']}' classified as: {label}")
        else:
            print("  → No entities found")
    
    print_step(2, "Classification Summary")
    print("\nEntity type distribution:")
    for label, count in sorted(entity_counts.items()):
        print(f"  {label}: {count} occurrences")

def demo_academic_document():
    """Demonstrate processing of academic document content"""
    print_header("ACADEMIC DOCUMENT PROCESSING DEMO")
    
    # Sample academic document content
    document_content = """
    COMPUTER SCIENCE PROGRAM STRUCTURE
    
    The Computer Science program is designed as an 8-semester curriculum.
    
    FIRST SEMESTER:
    - Introduction to Programming
    - Mathematics I  
    - English Composition
    
    SECOND SEMESTER:
    - Data Structures
    - Mathematics II
    - Physics I
    
    SPRING SEMESTER OFFERINGS:
    - Advanced Programming
    - Database Systems
    - Software Engineering
    
    FALL SEMESTER SCHEDULE:
    - Algorithms and Complexity
    - Computer Networks
    - Operating Systems
    
    SUMMER SEMESTER:
    - Internship Program
    - Special Projects
    
    Students must complete all semester requirements for graduation.
    The academic year is divided into fall semester and spring semester.
    Each semester has specific prerequisites and credit requirements.
    """
    
    ner_system = SimpleSemesterNER()
    
    print_step(1, "Processing Academic Document")
    print("Document content (sample):")
    print(document_content[:200] + "...")
    
    # Split into sentences for processing
    sentences = [s.strip() for s in document_content.split('.') if s.strip() and len(s.strip()) > 10]
    
    print(f"\nExtracted {len(sentences)} sentences from document")
    
    print_step(2, "Analyzing Sentences for Semester Information")
    
    semester_sentences = []
    
    for sentence in sentences:
        result = ner_system.process_text(sentence)
        if result['entities']:
            semester_sentences.append(result)
    
    print(f"\nFound {len(semester_sentences)} sentences with semester information:")
    print("-" * 60)
    
    for i, result in enumerate(semester_sentences, 1):
        print(f"\n{i}. \"{result['text']}\"")
        for entity in result['entities']:
            print(f"   → {entity['text']} ({entity['label']})")
    
    # Summary statistics
    stats = ner_system.get_statistics()
    coverage = len(semester_sentences) / len(sentences) * 100
    
    print_step(3, "Document Analysis Summary")
    print(f"Total sentences: {len(sentences)}")
    print(f"Sentences with semester info: {len(semester_sentences)}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"Total entities found: {stats['entities_found']}")

def test_system_functionality():
    """Test basic system functionality"""
    print_header("SYSTEM FUNCTIONALITY TEST")
    
    try:
        print_step(1, "Testing Core Components")
        
        # Test pattern recognition
        processor = SimpleTextProcessor()
        test_text = "This is the first semester of computer science."
        entities = processor.extract_semester_entities(test_text)
        
        if entities:
            print("✅ Pattern recognition working!")
            print(f"   Found: {entities[0]['text']} → {entities[0]['label']}")
        else:
            print("❌ Pattern recognition failed!")
            return False
        
        # Test NER system
        print_step(2, "Testing NER System")
        ner = SimpleSemesterNER()
        result = ner.process_text("Students study in fall semester.")
        
        if result['entities']:
            print("✅ NER system working!")
            print(f"   Processed: {result['entity_count']} entities")
        else:
            print("❌ NER system failed!")
            return False
        
        # Test batch processing
        print_step(3, "Testing Batch Processing")
        batch_texts = [
            "First semester courses are fundamental.",
            "Spring semester registration opens soon."
        ]
        
        batch_results = ner.batch_process(batch_texts)
        total_entities = sum(r['entity_count'] for r in batch_results)
        
        if total_entities > 0:
            print("✅ Batch processing working!")
            print(f"   Processed {len(batch_texts)} texts, found {total_entities} entities")
        else:
            print("❌ Batch processing failed!")
            return False
        
        print_step(4, "All Tests Passed!")
        print("✅ System is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {str(e)}")
        return False

def main():
    """Run the simple demo"""
    print_header("SEMESTER NER SYSTEM - SIMPLE DEMO")
    print("This demo tests core functionality without external dependencies")
    print("Perfect for verifying the system works before full setup!")
    
    try:
        # Run demo components
        print("\n🚀 Starting demonstrations...")
        
        # Test 1: System functionality
        if not test_system_functionality():
            print("❌ Basic functionality test failed!")
            return False
        
        # Test 2: Pattern matching
        demo_pattern_matching()
        
        # Test 3: Entity classification  
        demo_entity_classification()
        
        # Test 4: Academic document processing
        demo_academic_document()
        
        # Success summary
        print_header("DEMO COMPLETED SUCCESSFULLY!")
        print("✅ All core components are working correctly!")
        print("\nWhat this demo showed:")
        print("• Pattern-based semester entity recognition")
        print("• Text processing and classification")
        print("• Batch processing capabilities")
        print("• Academic document analysis")
        
        print(f"\nNext steps:")
        print("1. Run full setup: python setup.py")
        print("2. Install Tesseract OCR for PDF processing")
        print("3. Try the complete system: python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("This indicates there may be basic Python issues.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Demo completed successfully!")
    else:
        print("\n💥 Demo failed - check the errors above")
    
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1) 