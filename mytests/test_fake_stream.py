def stream():
    """Stream response with learning after completion."""
    chunks = ["Hello", "You", "are the", "best", "student", "ever", "!"]

    full_response = []
    for chunk in chunks:
        full_response.append(chunk)
        yield chunk  # Stream to caller

    # Learn after stream completes
    complete_response = " ".join(full_response)
    print(f"\n\nComplete response: {complete_response}")


if __name__ == "__main__":
    print("Streaming response:\n")
    for chunk in stream():
        print(chunk, end=" ", flush=True)
