document.querySelector("form").addEventListener("submit", function(e) {
    e.preventDefault();
    predict();
});

async function predict() {
    const message = document.getElementById("message").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    });

    const data = await response.json();

    if (data.detail) {
        document.getElementById("result").innerHTML = "Error: " + data.detail;
        return  ;
    }

    // Default probability
    let probability_percent = "N/A";

    // If model supports predict_proba
    if (data.probabilities && data.probabilities.length > 1) {
        probability_percent = (data.probabilities[0] * 100).toFixed(1);
    }

    document.getElementById("result").innerHTML =
        `Message is <b>${data.prediction}</b> and I am ${probability_percent}% confident`;
        
}
