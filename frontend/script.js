document.getElementById('predictor-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  
  // Show calculating state
  const wrapper = document.getElementById('results-wrapper');
  const badge = document.getElementById('result-badge');
  const probVal = document.getElementById('prob-val');
  const probFill = document.getElementById('probability-fill');
  
  wrapper.classList.remove('hidden');
  badge.className = "result-badge";
  badge.style.background = "rgba(255, 255, 255, 0.05)";
  badge.style.color = "var(--text-secondary)";
  badge.innerText = "Calculating...";
  probVal.innerText = "0.00%";
  probFill.style.width = "0%";
  
  const payload = {
    Pregnancies: parseInt(document.getElementById('Pregnancies').value),
    Glucose: parseInt(document.getElementById('Glucose').value),
    BloodPressure: parseInt(document.getElementById('BloodPressure').value),
    SkinThickness: parseInt(document.getElementById('SkinThickness').value),
    Insulin: parseInt(document.getElementById('Insulin').value),
    BMI: parseFloat(document.getElementById('BMI').value),
    DiabetesPedigreeFunction: parseFloat(document.getElementById('DiabetesPedigreeFunction').value),
    Age: parseInt(document.getElementById('Age').value)
  };

  try {
    // Call the localhost backend API
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Update Badge
    badge.innerText = data.prediction;
    if (data.prediction === "Diabetic") {
      badge.className = "result-badge badge-diabetic";
    } else {
      badge.className = "result-badge badge-nondiabetic";
    }
    
    // Update probability
    const percentage = (data.probability * 100).toFixed(1);
    probVal.innerText = `${percentage}%`;
    probFill.style.width = `${percentage}%`;
    
  } catch (err) {
    badge.innerText = "Error";
    badge.style.background = "var(--danger-gradient)";
    badge.style.color = "#fff";
    probVal.innerText = "Failed to fetch prediction";
    console.error(err);
  }
});
