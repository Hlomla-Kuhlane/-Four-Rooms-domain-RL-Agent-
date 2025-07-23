# RL Assignment - Four Rooms Package Collection

1. Install: `pip install -r requirements.txt`
2. Run:
   
   python Scenario1.py          # Simple (1 package)
   python Scenario2.py          # Multi (4 packages) 
   python Scenario3.py          # Ordered (R→G→B)
Add -stochastic for 20% action failure

Key Files
Scenario1.py: 1 package collection

Scenario2.py: 4 package collection

Scenario3.py: Ordered R→G→B collection

FourRooms.py: Environment (don't modify)

Features
Q-learning implementation

Two exploration strategies (Scenario 1)

Stochastic action support

Automatic visualization:

output/scenario*_path.png

output/scenario*_rewards.png

Options
bash
-episodes NUM    # Training episodes (default: 500)
-boltzmann       # Use Boltzmann exploration (Scenario 1)
-stochastic      # Enable 20% action failure

