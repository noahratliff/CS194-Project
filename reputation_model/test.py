# test_model_suitability.py
import unittest
from model_suitability import ModelSuitability

class TestModelSuitability(unittest.TestCase):
    def setUp(self):
        # Initialize the ModelSuitability with SentenceTransformer embeddings
        self.model_suitability = ModelSuitability(model_name='all-MiniLM-L6-v2')
        
        # Define medical and legal questions
        self.medical_questions = [
            "What are common symptoms of the flu?",
            "How is diabetes diagnosed?",
            "What are the side effects of aspirin?",
            "Can high blood pressure cause headaches?",
            "How do vaccines work?"
        ]
        
        self.legal_questions = [
            "What is the statute of limitations for a contract?",
            "How is negligence proven in court?",
            "What rights do tenants have in eviction cases?",
            "What are the legal requirements for a will?",
            "What is intellectual property law?"
        ]
        
        # Simulate 80% success on medical questions
        for question in self.medical_questions[:4]:  # 4 correct
            self.model_suitability.update(question, success=True)
        self.model_suitability.update(self.medical_questions[4], success=False)  # 1 incorrect
        
        # Simulate 20% success on legal questions
        self.model_suitability.update(self.legal_questions[0], success=True)  # 1 correct
        for question in self.legal_questions[1:]:  # 4 incorrect
            self.model_suitability.update(question, success=False)
    
    def test_suitability_score_medical(self):
        # Test suitability score for a new medical question
        new_medical_question = "What treatments are available for hypertension?"
        medical_score = self.model_suitability.suitability_score(new_medical_question)
        
        print(f"Medical suitability score: {medical_score}")
        # Assert medical score is high
        self.assertGreater(medical_score, 0.5, "Expected higher suitability score for medical question")
    
    def test_suitability_score_legal(self):
        # Test suitability score for a new legal question
        new_legal_question = "What is required for a contract to be valid?"
        legal_score = self.model_suitability.suitability_score(new_legal_question)
        
        print(f"Legal suitability score: {legal_score}")
        # Assert legal score is lower than medical score
        self.assertLess(legal_score, 0.5, "Expected lower suitability score for legal question")
    
    def test_suitability_comparison(self):
        # Test both legal and medical question scores and compare
        new_medical_question = "What treatments are available for hypertension?"
        new_legal_question = "What is required for a contract to be valid?"
    
        medical_score = self.model_suitability.suitability_score(new_medical_question)
        legal_score = self.model_suitability.suitability_score(new_legal_question)
        
        print(f"Medical suitability score: {medical_score}")
        print(f"Legal suitability score: {legal_score}")
        # Assert medical score is higher than legal score
        self.assertGreater(medical_score, legal_score, "Expected higher suitability score for medical question over legal question")
    
if __name__ == "__main__":
    unittest.main()
