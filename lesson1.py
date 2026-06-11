import random
from collections import defaultdict

# Tiny "policy": for each question, store scores for candidate answers
policy = defaultdict(lambda: defaultdict(float))

def question():
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    return f"{a}+{b}", a + b

def generate_answer(q):
    # Candidate answers from 0..20
    candidates = list(range(21))
    scores = [policy[q][c] for c in candidates]

    # Soft preference: mostly exploit, sometimes explore
    if random.random() < 0.2:
        return random.choice(candidates)

    max_score = max(scores)
    best = [c for c in candidates if policy[q][c] == max_score]
    return random.choice(best)

def reward(predicted, correct):
    return 1.0 if predicted == correct else -0.1

def train(steps=5000, lr=0.1):
    for step in range(steps):
        q, correct = question()
        answer = generate_answer(q)
        r = reward(answer, correct)

        # Reinforce good answers, penalize bad ones
        policy[q][answer] += lr * r

def test(n=20):
    correct_count = 0

    for _ in range(n):
        q, correct = question()
        answer = generate_answer(q)

        if answer == correct:
            correct_count += 1

        print(f"{q} -> model: {answer}, correct: {correct}")

    print(f"\nAccuracy: {correct_count}/{n}")


if __name__ == "__main__":
    train()
    test()