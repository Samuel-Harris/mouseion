const $ = (id) => document.getElementById(id);

function tags(value) {
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function setTags(input, values) {
  input.value = values.join(", ");
}

function domainTag(value) {
  try {
    const parsed = new URL(normalizedUrl(value));
    return parsed.hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return "";
  }
}

function normalizedUrl(value) {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error("Enter a URL first");
  }
  return trimmed.includes("://") ? trimmed : `https://${trimmed}`;
}

function syncUrlDomainTag() {
  const input = $("url-input");
  const tagInput = $("url-tags");
  const previous = input.dataset.domainTag || "";
  const next = domainTag(input.value);
  const current = tags(tagInput.value).filter(
    (tag) => tag.toLowerCase() !== previous.toLowerCase()
  );
  if (next && !current.some((tag) => tag.toLowerCase() === next)) {
    current.push(next);
  }
  setTags(tagInput, current);
  input.dataset.domainTag = next;
}

function setButtonBusy(button, label) {
  button.disabled = true;
  button.dataset.defaultLabel = button.textContent;
  button.textContent = label;
}

function clearButtonBusy(button) {
  button.disabled = false;
  button.textContent = button.dataset.defaultLabel || button.textContent;
  delete button.dataset.defaultLabel;
}

function showMessage(text, isError = false) {
  const node = $("message");
  node.textContent = text;
  node.className = isError ? "error" : "ok";
}

function ingestMessage(result, label = result.title) {
  const chunks = `${result.chunks_created} chunks`;
  if (result.action === "replaced") {
    return `Collision: replaced existing ${label}: ${chunks}`;
  }
  return `Ingested ${label}: ${chunks}`;
}

function selectedFile() {
  const input = $("file-input");
  return input.files.length ? input.files[0] : null;
}

function updateFileLabel() {
  const file = selectedFile();
  $("file-label").textContent = file
    ? file.name
    : "Drop or choose PDF, Markdown, text, or HTML";
}

function clearSelectedFile() {
  $("file-input").value = "";
  updateFileLabel();
}

function preventFileNavigation(event) {
  event.preventDefault();
}

function enableFileDropzone() {
  const dropzone = document.querySelector(".dropzone");
  const input = $("file-input");

  window.addEventListener("dragover", preventFileNavigation);
  window.addEventListener("drop", preventFileNavigation);

  input.addEventListener("change", updateFileLabel);

  dropzone.addEventListener("dragenter", () => {
    dropzone.classList.add("is-dragging");
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
  dropzone.addEventListener("dragleave", (event) => {
    if (!dropzone.contains(event.relatedTarget)) {
      dropzone.classList.remove("is-dragging");
    }
  });
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
    if (!event.dataTransfer?.files.length) {
      return;
    }
    input.files = event.dataTransfer.files;
    updateFileLabel();
  });
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "content-type": "application/json", ...(options.headers || {}) },
  });
  const text = await response.text();
  let body = {};
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text };
    }
  }
  if (!response.ok) {
    throw new Error(body.detail || body.error || "Request failed");
  }
  return body;
}

async function refreshStats() {
  const stats = await jsonFetch("/api/stats");
  $("stats").textContent = `${stats.documents} documents, ${stats.chunks} chunks`;
}

function renderResults(results) {
  const root = $("results");
  root.innerHTML = "";
  if (!results.length) {
    root.innerHTML = '<p class="empty">No confident results.</p>';
    return;
  }
  for (const result of results) {
    const article = document.createElement("article");
    article.className = "result";
    const document = result.document;
    const metadata = document.metadata || {};
    const sourceUrl = metadata.html_url || metadata.pdf_url || "";
    const source = sourceUrl
      ? `<a href="${escapeAttribute(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(document.source)}</a>`
      : escapeHtml(document.source);
    const tags = (document.tags || []).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
    const authors = metadata.authors ? `<p class="byline">${escapeHtml(metadata.authors)}</p>` : "";
    const updated = metadata.update_date ? ` · updated ${escapeHtml(metadata.update_date)}` : "";
    article.innerHTML = `
      <h3>${escapeHtml(document.title)}</h3>
      ${authors}
      <p class="meta">${source} · ${escapeHtml(document.type)} · ${escapeHtml(result.match || "match")} · score ${Number(result.score).toFixed(3)}${updated}</p>
      <div class="tags">${tags}</div>
      <p>${escapeHtml(result.content).slice(0, 700)}</p>
    `;
    root.appendChild(article);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char];
  });
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

$("refresh").addEventListener("click", refreshStats);
$("url-input").addEventListener("input", syncUrlDomainTag);

enableFileDropzone();

$("url-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter || $("url-form").querySelector("button[type='submit']");
  try {
    syncUrlDomainTag();
    setButtonBusy(button, "Ingesting...");
    showMessage("Ingesting URL...");
    const body = {
      url: normalizedUrl($("url-input").value),
      tags: tags($("url-tags").value),
    };
    const result = await jsonFetch("/api/url", { method: "POST", body: JSON.stringify(body) });
    showMessage(ingestMessage(result));
    await refreshStats();
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    clearButtonBusy(button);
  }
});

$("memory-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const body = {
      content: $("memory-input").value,
      tags: tags($("memory-tags").value),
    };
    const result = await jsonFetch("/api/memory", { method: "POST", body: JSON.stringify(body) });
    if (result.action === "replaced") {
      showMessage(`Collision: replaced existing memory ${result.memory_id}`);
    } else {
      showMessage(`Saved memory ${result.memory_id}`);
    }
    await refreshStats();
  } catch (error) {
    showMessage(error.message, true);
  }
});

$("file-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const file = selectedFile();
    if (!file) {
      throw new Error("Choose a file first");
    }
    const form = new FormData();
    form.append("file", file);
    form.append("tags", $("file-tags").value);
    const response = await fetch("/api/files", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || result.error || "Upload failed");
    }
    showMessage(ingestMessage(result));
    clearSelectedFile();
    await refreshStats();
  } catch (error) {
    showMessage(error.message, true);
  }
});

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const body = {
      query: $("search-input").value,
      top_k: Number($("search-top-k").value),
      search_syntax: $("search-advanced").checked ? "advanced" : "plain",
    };
    const type = $("search-type").value;
    const tagValues = tags($("search-tags").value);
    if (type !== "all" || tagValues.length) {
      body.filter = {};
      if (type !== "all") {
        body.filter.type = type;
      }
      if (tagValues.length) {
        body.filter.tags = tagValues;
      }
    }
    const result = await jsonFetch("/api/search", { method: "POST", body: JSON.stringify(body) });
    renderResults(result.results);
    showMessage(result.message || `${result.results.length} results`);
  } catch (error) {
    showMessage(error.message, true);
  }
});

refreshStats().catch((error) => showMessage(error.message, true));
