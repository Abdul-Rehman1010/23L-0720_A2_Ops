// Convenience only: the server repeats every check, so nothing here is trusted.
(function () {
  var form = document.getElementById("upload-form");
  if (!form) return;
  var input = document.getElementById("image");
  var preview = document.getElementById("preview");
  var title = document.getElementById("dz-title");
  var msg = document.getElementById("client-msg");
  var button = document.getElementById("submit-btn");
  var zone = document.getElementById("dropzone");
  var maxBytes = parseInt(form.dataset.maxBytes, 10);

  function showMessage(text) {
    msg.textContent = text;
    msg.hidden = !text;
  }

  input.addEventListener("change", function () {
    showMessage("");
    var file = input.files && input.files[0];
    if (!file) { preview.hidden = true; title.textContent = "Choose an image"; return; }
    title.textContent = file.name;
    if (file.size > maxBytes) {
      showMessage("This file is " + (file.size / 1048576).toFixed(1) + " MB. The limit is " + (maxBytes / 1048576).toFixed(0) + " MB.");
    }
    if (file.type.indexOf("image/") === 0) {
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
    } else {
      preview.hidden = true;
    }
  });

  form.addEventListener("submit", function () {
    button.disabled = true;
    button.textContent = "Analysing…";
  });

  // coming back with the browser's Back button must not leave the button stuck on "Analysing…"
  window.addEventListener("pageshow", function () {
    button.disabled = false;
    button.textContent = "Analyse image";
  });

  ["dragenter", "dragover"].forEach(function (name) {
    zone.addEventListener(name, function (e) { e.preventDefault(); zone.classList.add("dragover"); });
  });
  ["dragleave", "drop"].forEach(function (name) {
    zone.addEventListener(name, function (e) { e.preventDefault(); zone.classList.remove("dragover"); });
  });
  zone.addEventListener("drop", function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      input.dispatchEvent(new Event("change"));
    }
  });
})();
