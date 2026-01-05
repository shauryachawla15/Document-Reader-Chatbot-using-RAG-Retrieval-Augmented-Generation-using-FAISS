function ask() {
    const question = document.getElementById("question").value;

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
    })
        .then(res => res.json())
        .then(data => {
            document.getElementById("response").innerText =
                data.answer + "\n\n" + data.context.join("\n\n");
        });
}
