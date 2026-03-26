from candidate_module import compute_score

def main():
    test_input = 10.0
    score = compute_score(test_input)
    print({"score": score})

if __name__ == "__main__":
    main()
