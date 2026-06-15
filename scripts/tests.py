from tennis_functions import reorder_participants_alphabetically
import pandas as pd

def reorder_participants_alphabetically_test():
    """
    Unit test for reorder_participants_alphabetically function.
    Tests three scenarios:
    1. No swap needed (already alphabetical)
    2. Swap needed (participant1 surname > participant2 surname)
    3. Verification that all columns are swapped correctly
    
    Returns:
    --------
    bool : True if all tests pass

    Written with Claude - prompt "Write a unit test for reorder_participants_alphabetically with three known outomces; one where no swap is needed, and two where it is. Assert that 
    all statistics have been swapped correctly."
    """
    
    # Create test dataset with known outcomes
    test_data = pd.DataFrame({
        'start_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'tournament_name': ['Test Open', 'Test Open', 'Test Open'],
        'surface': ['Hard', 'Hard', 'Hard'],
        'best_of': [3, 3, 3],
        'round_name': ['Final', 'Final', 'Final'],
        
        # Case 1: Federer (F) < Nadal (N) - NO SWAP
        # Case 2: Nadal (N) > Djokovic (D) - SWAP
        # Case 3: Murray (M) > Alcaraz (A) - SWAP
        'participant1_name': ['Roger Federer', 'Rafael Nadal', 'Andy Murray'],
        'participant2_name': ['Rafael Nadal', 'Novak Djokovic', 'Carlos Alcaraz'],
        
        'participant1_games_won': [12, 13, 14],
        'participant1_sets_won': [2, 2, 2],
        'participant1_is_home': [1, 0, 1],
        'participant1_odds': [1.5, 2.0, 1.8],
        
        'participant2_games_won': [10, 11, 12],
        'participant2_sets_won': [1, 0, 1],
        'participant2_is_home': [0, 1, 0],
        'participant2_odds': [2.5, 1.9, 2.2],
    })
    
    ("UNIT TEST: reorder_participants_alphabetically")
     # Apply function
    result = reorder_participants_alphabetically(test_data)
    
    # Run assertions
    try:
        # Case 1: No swap (Federer < Nadal alphabetically)
        assert result.iloc[0]['participant1_name'] == 'Roger Federer'
        assert result.iloc[0]['participant1_games_won'] == 12
        assert result.iloc[0]['participant1_odds'] == 1.5
        assert result.iloc[0]['is_participant1_winner'] == 1
        print("Case 1 PASSED: Federer vs Nadal (no swap)")
        
        # Case 2: Swap needed (Djokovic < Nadal alphabetically)
        assert result.iloc[1]['participant1_name'] == 'Novak Djokovic'
        assert result.iloc[1]['participant2_name'] == 'Rafael Nadal'
        assert result.iloc[1]['participant1_games_won'] == 11  # Originally p2
        assert result.iloc[1]['participant2_games_won'] == 13  # Originally p1
        assert result.iloc[1]['participant1_odds'] == 1.9      # Originally p2
        assert result.iloc[1]['participant2_odds'] == 2.0      # Originally p1
        assert result.iloc[1]['participant1_is_home'] == 1     # Originally p2
        assert result.iloc[1]['participant2_is_home'] == 0     # Originally p1
        assert result.iloc[1]['is_participant1_winner'] == 0   # Djokovic lost
        print("Case 2 PASSED: Nadal vs Djokovic (swap)")
        
        # Case 3: Swap needed (Alcaraz < Murray alphabetically)
        assert result.iloc[2]['participant1_name'] == 'Carlos Alcaraz'
        assert result.iloc[2]['participant2_name'] == 'Andy Murray'
        assert result.iloc[2]['participant1_games_won'] == 12  # Originally p2
        assert result.iloc[2]['participant2_games_won'] == 14  # Originally p1
        assert result.iloc[2]['is_participant1_winner'] == 0   # Alcaraz lost
        print("Case 3 PASSED: Murray vs Alcaraz (swap)")
        
        print("ALL TESTS PASSED ✓")
        return True
        
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        print("\nExpected vs Actual:")
        print(result[['participant1_name', 'participant2_name', 
                      'participant1_games_won', 'participant2_games_won',
                      'is_participant1_winner']])
        return False

# Run the test
reorder_participants_alphabetically_test()
