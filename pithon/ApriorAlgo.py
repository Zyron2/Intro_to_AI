import itertools
from typing import List, Set, Tuple, Dict
from collections import defaultdict

# ============================================================================
# APRIORI ALGORITHM IMPLEMENTATION
# ============================================================================

class AprioriAlgorithm:
    """
    Apriori algorithm for frequent itemset mining and association rule learning.
    """
    
    def __init__(self, transactions: List[Set], min_support: float = 0.2):
        """
        Initialize Apriori algorithm.
        
        Args:
            transactions: List of transactions (each transaction is a set of items)
            min_support: Minimum support threshold (0-1)
        """
        self.transactions = transactions
        self.min_support = min_support
        self.total_transactions = len(transactions)
        self.frequent_itemsets = {}
        self.association_rules = []
    
    def calculate_support(self, itemset: frozenset) -> float:
        """Calculate support for an itemset."""
        count = sum(1 for transaction in self.transactions if itemset.issubset(transaction))
        return count / self.total_transactions
    
    def get_candidates(self, level: int, prev_itemsets: List[frozenset]) -> List[frozenset]:
        """Generate candidate itemsets from previous level."""
        candidates = set()
        prev_list = list(prev_itemsets)
        
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                union = prev_list[i] | prev_list[j]
                if len(union) == level:
                    candidates.add(union)
        
        return list(candidates)
    
    def find_frequent_itemsets(self) -> Dict[int, List[Tuple[frozenset, float]]]:
        """Find all frequent itemsets using Apriori algorithm."""
        # Get unique items
        all_items = set()
        for transaction in self.transactions:
            all_items.update(transaction)
        
        # Level 1: Single items
        level = 1
        current_candidates = [frozenset([item]) for item in all_items]
        
        while current_candidates:
            frequent = []
            
            for candidate in current_candidates:
                support = self.calculate_support(candidate)
                if support >= self.min_support:
                    frequent.append((candidate, support))
            
            if not frequent:
                break
            
            self.frequent_itemsets[level] = frequent
            
            # Generate next level candidates
            if level > 1 or len(frequent) > 1:
                current_candidates = self.get_candidates(level + 1, [item[0] for item in frequent])
            else:
                current_candidates = []
            
            level += 1
        
        return self.frequent_itemsets
    
    def generate_rules(self, min_confidence: float = 0.6) -> List[Dict]:
        """
        Generate association rules from frequent itemsets.
        
        Args:
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            List of association rules
        """
        rules = []
        
        # Get itemsets with 2 or more items
        for level in sorted(self.frequent_itemsets.keys()):
            if level < 2:
                continue
            
            for itemset, support in self.frequent_itemsets[level]:
                # Generate all possible rules from this itemset
                items = list(itemset)
                
                for r in range(1, len(items)):
                    for antecedent_tuple in itertools.combinations(items, r):
                        antecedent = frozenset(antecedent_tuple)
                        consequent = itemset - antecedent
                        
                        # Calculate support of antecedent
                        antecedent_support = self.calculate_support(antecedent)
                        
                        if antecedent_support == 0:
                            continue
                        
                        # Calculate confidence
                        confidence = support / antecedent_support
                        
                        if confidence >= min_confidence:
                            # Calculate lift
                            consequent_support = self.calculate_support(consequent)
                            lift = confidence / consequent_support if consequent_support > 0 else 0
                            
                            rule = {
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            }
                            
                            # Avoid duplicates
                            if rule not in rules:
                                rules.append(rule)
        
        self.association_rules = rules
        return rules
    
    def display_results(self):
        """Display algorithm results in a formatted way."""
        print("\n" + "="*80)
        print("FREQUENT ITEMSETS")
        print("="*80)
        
        for level in sorted(self.frequent_itemsets.keys()):
            print(f"\n{level}-Itemsets (Support ≥ {self.min_support}):")
            print("-" * 80)
            
            for itemset, support in self.frequent_itemsets[level]:
                items_str = ", ".join(sorted(list(itemset)))
                print(f"   {{{items_str}:15}} → Support: {support:.2%}")
    
    def display_rules_as_table(self, rules):
        """Display association rules in table format."""
        if not rules:
            print("No association rules found with the given confidence threshold.")
            return
        
        # Sort rules by confidence (highest first)
        rules_sorted = sorted(rules, key=lambda x: x['confidence'], reverse=True)
        
        # Calculate column widths
        rule_col_width = 50
        confidence_col_width = 15
        support_col_width = 12
        lift_col_width = 10
        
        # Print table header
        print("\n" + "─" * (rule_col_width + confidence_col_width + support_col_width + lift_col_width + 8))
        print(f"{'RULE':<{rule_col_width}} │ {'CONFIDENCE':<{confidence_col_width}} │ {'SUPPORT':<{support_col_width}} │ {'LIFT':<{lift_col_width}}")
        print("─" * (rule_col_width + confidence_col_width + support_col_width + lift_col_width + 8))
        
        # Print rules
        for rule in rules_sorted:
            antecedent_str = ", ".join(sorted(list(rule['antecedent'])))
            consequent_str = ", ".join(sorted(list(rule['consequent'])))
            
            rule_str = f"{{{antecedent_str}}} → {{{consequent_str}}}"
            confidence_str = f"{rule['confidence']:.2%}"
            support_str = f"{rule['support']:.2%}"
            lift_str = f"{rule['lift']:.2f}"
            
            print(f"{rule_str:<{rule_col_width}} │ {confidence_str:>{confidence_col_width}} │ {support_str:>{support_col_width}} │ {lift_str:>{lift_col_width}}")
        
        print("─" * (rule_col_width + confidence_col_width + support_col_width + lift_col_width + 8))


# ============================================================================
# EXAMPLE 1: PHARMACY PURCHASE ANALYSIS
# ============================================================================

class PharmacyAnalysis:
    """Analyzes purchase patterns in pharmacy transactions."""
    
    @staticmethod
    def run_example():
        """Run pharmacy purchase analysis example."""
        print("\n" + "="*80)
        print(" "*15 + "EXAMPLE 1: PHARMACY PURCHASE PATTERN ANALYSIS")
        print("="*80)
        
        # Sample pharmacy transactions
        # Each transaction is a set of items purchased by a customer
        transactions = [
            {'Aspirin', 'Band-Aids', 'Thermometer'},
            {'Cough Syrup', 'Tissue', 'Vitamin C'},
            {'Aspirin', 'Ibuprofen', 'Water Bottle'},
            {'Band-Aids', 'Antiseptic', 'Thermometer'},
            {'Aspirin', 'Band-Aids'},
            {'Cough Syrup', 'Vitamin C', 'Honey'},
            {'Aspirin', 'Ibuprofen'},
            {'Band-Aids', 'Thermometer', 'Antiseptic'},
            {'Cough Syrup', 'Tissue', 'Vitamin C'},
            {'Aspirin', 'Thermometer'},
            {'Vitamin C', 'Honey', 'Tissue'},
            {'Ibuprofen', 'Water Bottle'},
            {'Aspirin', 'Band-Aids', 'Ibuprofen'},
            {'Cough Syrup', 'Tissue'},
            {'Thermometer', 'Antiseptic'},
        ]
        
        # Convert to frozensets
        transactions_set = [set(t) for t in transactions]
        
        # Run Apriori algorithm
        apriori = AprioriAlgorithm(transactions_set, min_support=0.20)
        apriori.find_frequent_itemsets()
        rules = apriori.generate_rules(min_confidence=0.50)
        
        # Display results
        print(f"\n📊 TRANSACTION SUMMARY:")
        print(f"   • Total transactions: {apriori.total_transactions}")
        print(f"   • Minimum support: {apriori.min_support:.0%}")
        print(f"   • Minimum confidence: 50%")
        
        apriori.display_results()
        
        # Display association rules
        print("\n" + "="*80)
        print("ASSOCIATION RULES")
        print("="*80)
        print(f"\nRules found: {len(rules)}")
        
        apriori.display_rules_as_table(rules)
        
        # Insight
        print("\n" + "="*80)
        print("💡 BUSINESS INSIGHTS:")
        print("="*80)
        print("""
   1. Customers who buy Aspirin often also buy Band-Aids (treating minor injuries)
   2. Customers who buy Cough Syrup frequently purchase Tissue and Vitamin C
   3. Thermometer, Antiseptic, and Band-Aids form a common illness care bundle
   4. These insights can help with store layout and promotional bundling
        """)


# ============================================================================
# EXAMPLE 2: MOVIE STREAMING PLATFORM ANALYSIS
# ============================================================================

class MovieStreamingAnalysis:
    """Analyzes viewing patterns on a movie streaming platform."""
    
    @staticmethod
    def run_example():
        """Run movie streaming analysis example."""
        print("\n" + "="*80)
        print(" "*12 + "EXAMPLE 2: MOVIE STREAMING PLATFORM VIEWING PATTERN ANALYSIS")
        print("="*80)
        
        # Sample streaming transactions
        # Each transaction represents movies watched by a user in a session
        transactions = [
            {'Inception', 'The Matrix', 'Interstellar'},
            {'The Avengers', 'Iron Man', 'Captain America'},
            {'Inception', 'The Matrix'},
            {'The Lion King', 'Frozen', 'Aladdin'},
            {'Inception', 'Interstellar', 'Dune'},
            {'The Avengers', 'Iron Man'},
            {'The Lion King', 'Frozen'},
            {'The Matrix', 'Interstellar'},
            {'The Avengers', 'Captain America'},
            {'Frozen', 'Aladdin'},
            {'Inception', 'Dune'},
            {'Iron Man', 'Captain America'},
            {'The Lion King', 'Aladdin'},
            {'The Matrix', 'Dune'},
            {'The Avengers', 'Iron Man', 'Captain America'},
            {'Inception', 'The Matrix', 'Dune'},
            {'Frozen', 'Aladdin', 'The Lion King'},
            {'Interstellar', 'Dune'},
            {'The Avengers', 'Captain America'},
            {'Inception', 'Interstellar'},
        ]
        
        # Convert to frozensets
        transactions_set = [set(t) for t in transactions]
        
        # Run Apriori algorithm
        apriori = AprioriAlgorithm(transactions_set, min_support=0.20)
        apriori.find_frequent_itemsets()
        rules = apriori.generate_rules(min_confidence=0.50)
        
        # Display results
        print(f"\n📊 VIEWING SESSION SUMMARY:")
        print(f"   • Total viewing sessions: {apriori.total_transactions}")
        print(f"   • Minimum support: {apriori.min_support:.0%}")
        print(f"   • Minimum confidence: 50%")
        
        apriori.display_results()
        
        # Display association rules
        print("\n" + "="*80)
        print("ASSOCIATION RULES (VIEWING PATTERNS)")
        print("="*80)
        print(f"\nRules found: {len(rules)}")
        
        apriori.display_rules_as_table(rules)
        
        # Insight
        print("\n" + "="*80)
        print("💡 PLATFORM INSIGHTS:")
        print("="*80)
        print("""
   1. Users who watch Sci-Fi movies (Inception, Matrix) often watch others in the genre
   2. Superhero fans (Avengers, Iron Man) tend to watch the entire franchise
   3. Disney animated movies (Frozen, Aladdin, Lion King) are watched together
   4. These patterns help with recommendation systems and content bundling
   5. Platform can suggest related movies based on viewing history
        """)


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main interface with user options."""
    while True:
        print("\n" + "="*80)
        print("APRIORI ALGORITHM - ASSOCIATION RULE LEARNING")
        print("="*80)
        print("\nChoose an example to view:")
        print("1. Example 1: Pharmacy Purchase Pattern Analysis")
        print("2. Example 2: Movie Streaming Platform Viewing Pattern Analysis")
        print("3. Run Both Examples")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            PharmacyAnalysis.run_example()
        elif choice == '2':
            MovieStreamingAnalysis.run_example()
        elif choice == '3':
            PharmacyAnalysis.run_example()
            MovieStreamingAnalysis.run_example()
        elif choice == '4':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1-4.")
        
        if choice in ['1', '2', '3']:
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
