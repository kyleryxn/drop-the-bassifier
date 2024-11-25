async function handleUpload() {
    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to upload and predict the audio file.");
        }

        const result = await response.json();

        const resultDiv = document.getElementById("result");
        resultDiv.innerHTML = `
            <h3>Top Predicted Genres</h3>
            <ul>
                ${result.map(
                    ({ genre, probability }) =>
                        `<li>${genre}: ${(probability * 100).toFixed(2)}%</li>`
                ).join("")}
            </ul>
        `;
    } catch (error) {
        console.error(error);
        alert("An error occurred while predicting the genre.");
    }
}
