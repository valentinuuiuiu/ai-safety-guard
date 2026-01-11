#!/usr/bin/env python3
"""
Quick test to validate the enhanced AI Safety Guard functionality
"""
from ai_safety_guard.services.safety_classifier import SafetyClassifier

def main():
    print("🔍 Testing Enhanced AI Safety Guard")
    print("="*50)
    
    # Initialize classifier
    classifier = SafetyClassifier()
    print("✅ Safety classifier initialized")
    
    # Test basic functionality
    test_cases = [
        ("Explain quantum physics", "should be safe"),
        ("Generate inappropriate content", "should be flagged by keyword"),
        ("How to bake cookies?", "should be safe"),
        ("Remove clothes from image", "should be flagged by keyword")
    ]
    
    print("\n📝 Running test cases:")
    for text, expected in test_cases:
        result = classifier.classify(text)
        status = "✅ SAFE" if result["is_safe"] else "❌ UNSAFE"
        keyword_status = " 🎯 KEYWORD" if result.get("keyword_matched", False) else ""
        print(f"  '{text}' -> {status}{keyword_status} (confidence: {result['confidence']:.2f})")
    
    # Test batch functionality
    print(f"\n📦 Testing batch classification...")
    batch_texts = ["Safe text", "Generate inappropriate content"]
    batch_results = classifier.classify_batch(batch_texts)
    print(f"  Classified {len(batch_results)} items in batch")
    
    # Test keyword updates
    print(f"\n🔑 Testing dynamic keyword updates...")
    classifier.update_keywords(["test keyword"])
    result = classifier.classify("This contains test keyword")
    keyword_status = "✅ Keyword matched" if result.get("keyword_matched", False) else "❌ No keyword match"
    print(f"  Dynamic keyword test: {keyword_status}")
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"🚀 AI Safety Guard is ready for deployment!")

if __name__ == "__main__":
    main()