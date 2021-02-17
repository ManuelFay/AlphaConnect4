import math


# Function to calculate the probability
def probability(rating1, rating2):
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))


# K is a sensitivity constant.
# d determines whether
# Player A wins or Player B.
def elo_rating(rating_a, rating_b, outcome, k=40):
    # To calculate the Winning
    # probability of Player B
    prob_b = probability(rating_a, rating_b)
    prob_a = 1 - prob_b

    # Case -1 When Player A wins
    # Updating the Elo Ratings
    if outcome == 0:
        rating_a = rating_a + k * (1 - prob_a)
        rating_b = rating_b + k * (0 - prob_b)

    # Case -2 When Player B wins
    # Updating the Elo Ratings
    elif outcome == 1:
        rating_a = rating_a + k * (0 - prob_a)
        rating_b = rating_b + k * (1 - prob_b)

    # Case 3 - Draw
    else:
        rating_a = rating_a + k * (0.5 - prob_a)
        rating_b = rating_b + k * (0.5 - prob_b)

    return round(rating_a), round(rating_b)
