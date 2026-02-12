// State
let state = { status: null, models: {}, backendState: null, loading: {} };
let previousState = { status: null, backendState: null, models: {}, loading: {} };
let chatMessages = [];
let isGenerating = false;
let abortController = null;
let currentModelId = null;
let eventSource = null;

// SSE connection for real-time updates
function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/api/events');

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'operation') {
                handleOperationEvent(data);
            } else if (data.type === 'state') {
                handleStateEvent(data);
            }
        } catch (e) {
            console.error('SSE parse error:', e);
        }
    };

    eventSource.onerror = () => {
        console.error('SSE connection lost, reconnecting in 3s...');
        eventSource.close();
        setTimeout(connectSSE, 3000);
    };
}

function handleStateEvent(data) {
    const oldActive = state.status?.active;
    const newActive = data.active;

    // Update state from SSE
    state.status = {
        active: data.active,
        backends: data.backends,
    };
    state.backendState = data.backendState;

    // Fetch models if active backend changed
    if (newActive !== oldActive) {
        fetchModels();
    }

    render();
}

function handleOperationEvent(event) {
    const { operation, status, backend, model, message, error } = event;

    // Normalize operation names: task:load_model -> model:load, task:download_model -> model:download
    const isModelOp = operation.startsWith('model:') || operation.startsWith('task:load_model') || operation.startsWith('task:download_model');
    const isLoadOp = operation === 'model:load' || operation === 'task:load_model';
    const isUnloadOp = operation === 'model:unload';

    if (status === 'started') {
        // Set loading state
        if (operation === 'backend:start' || operation === 'backend:stop') {
            state.loading[`backend-${backend}`] = true;
        } else if (isLoadOp) {
            state.loading[`model-${model}`] = true;
        } else if (isUnloadOp) {
            state.loading['unload'] = true;
        }
        render();
    } else if (status === 'completed') {
        // Clear loading, show success
        clearLoadingState(operation, backend, model);
        showToast(message || 'Completed', 'success');
        // Refresh models after model operations
        if (isModelOp) {
            fetchModels();
        }
        render();
    } else if (status === 'failed') {
        // Clear loading, show error
        clearLoadingState(operation, backend, model);
        showToast(error || 'Operation failed', 'error');
        render();
    }
}

function clearLoadingState(operation, backend, model) {
    if (operation === 'backend:start' || operation === 'backend:stop') {
        state.loading[`backend-${backend}`] = false;
    } else if (operation === 'model:load' || operation === 'task:load_model') {
        state.loading[`model-${model}`] = false;
    } else if (operation === 'model:unload') {
        state.loading['unload'] = false;
    }
}

// Fetch models separately (they change less frequently)
async function fetchModels() {
    if (!state.status?.active) {
        state.models = {};
        return;
    }
    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();
        state.models = data.models;
        renderModels();
    } catch (e) {
        console.error('Failed to fetch models:', e);
    }
}

// Legacy fetchAll for use after actions (models need refresh)
async function fetchAll() {
    await fetchModels();
}

function formatSize(bytes) {
    if (!bytes) return '';
    const gb = bytes / (1024 * 1024 * 1024);
    return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function deepEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}

function render() {
    // Selective rendering: only update sections that changed
    const statusChanged = !deepEqual(state.status, previousState.status);
    const backendStateChanged = !deepEqual(state.backendState, previousState.backendState);
    const modelsChanged = !deepEqual(state.models, previousState.models);
    const loadingChanged = !deepEqual(state.loading, previousState.loading);

    if (statusChanged || loadingChanged) {
        renderStatus();
        renderBackends();
    }
    if (backendStateChanged) {
        renderStatus();  // Status bar shows loaded model
        renderNodes();
        renderChatHeader();
    }
    if (modelsChanged || backendStateChanged || loadingChanged) {
        renderModels();
    }

    updateChatInput();

    // Save current state for next comparison
    previousState = {
        status: JSON.parse(JSON.stringify(state.status)),
        backendState: state.backendState ? JSON.parse(JSON.stringify(state.backendState)) : null,
        models: JSON.parse(JSON.stringify(state.models)),
        loading: JSON.parse(JSON.stringify(state.loading)),
    };
}

function renderStatus() {
    const el = document.getElementById('status-bar');
    if (state.status?.active) {
        const backend = state.status.backends[state.status.active];
        let text = `<span class="status-indicator active"></span>${backend.display_name}`;
        if (state.backendState?.loaded_model) {
            text += ` - ${state.backendState.loaded_model.name}`;
        }
        el.innerHTML = text;
    } else {
        el.innerHTML = '<span class="status-indicator inactive"></span>No backend running';
    }
}

function renderNodes() {
    const section = document.getElementById('nodes-section');
    const el = document.getElementById('nodes');

    if (!state.backendState?.nodes?.length) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    el.innerHTML = state.backendState.nodes.map(node => {
        const isOnline = node.status === 'online';
        const badge = isOnline
            ? '<span class="status-badge online">online</span>'
            : '<span class="status-badge offline">offline</span>';

        const memPct = node.memory_total ? Math.round(node.memory_used / node.memory_total * 100) : 0;
        const gpuMemPct = node.gpu_memory_total ? Math.round(node.gpu_memory_used / node.gpu_memory_total * 100) : 0;
        const gpuBusyPct = node.gpu_busy_percent || 0;
        const cpuPct = Math.round(node.cpu_percent || 0);
        const memClass = memPct > 90 ? 'high' : memPct > 70 ? 'medium' : '';
        const gpuMemClass = gpuMemPct > 90 ? 'high' : gpuMemPct > 70 ? 'medium' : '';
        const gpuBusyClass = gpuBusyPct > 90 ? 'high' : gpuBusyPct > 70 ? 'medium' : '';
        const cpuClass = cpuPct > 90 ? 'high' : cpuPct > 70 ? 'medium' : '';

        const hasGpu = node.gpu_name && node.gpu_name !== 'CPU';

        return `
            <div class="card ${isOnline ? 'active' : ''}">
                <div class="card-header">
                    <div class="card-info">
                        <h3>${node.hostname}${badge}</h3>
                        <div class="subtitle">${node.role} · ${node.gpu_name || 'CPU'}</div>
                    </div>
                </div>
                ${node.memory_total ? `
                <div class="node-stats">
                    <div class="stat-item">
                        <span class="stat-label">Mem</span>
                        <span class="stat-value">${formatSize(node.memory_used)} / ${formatSize(node.memory_total)}</span>
                        <div class="progress-bar"><div class="progress-fill ${memClass}" style="width: ${memPct}%"></div></div>
                    </div>
                    ${hasGpu && node.gpu_memory_total ? `
                    <div class="stat-item">
                        <span class="stat-label">GPU Mem</span>
                        <span class="stat-value">${formatSize(node.gpu_memory_used)} / ${formatSize(node.gpu_memory_total)}</span>
                        <div class="progress-bar"><div class="progress-fill ${gpuMemClass}" style="width: ${gpuMemPct}%"></div></div>
                    </div>
                    ` : ''}
                    ${hasGpu && node.gpu_name !== 'Metal' ? `
                    <div class="stat-item">
                        <span class="stat-label">GPU</span>
                        <span class="stat-value">${gpuBusyPct}%</span>
                        <div class="progress-bar"><div class="progress-fill ${gpuBusyClass}" style="width: ${gpuBusyPct}%"></div></div>
                    </div>
                    ` : ''}
                    <div class="stat-item">
                        <span class="stat-label">CPU</span>
                        <span class="stat-value">${cpuPct}%</span>
                        <div class="progress-bar"><div class="progress-fill ${cpuClass}" style="width: ${cpuPct}%"></div></div>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

function renderBackends() {
    const el = document.getElementById('backends');
    if (!state.status) return;

    // Check if any backend is currently loading
    const anyBackendLoading = Object.keys(state.status.backends).some(
        name => state.loading[`backend-${name}`]
    );

    el.innerHTML = Object.entries(state.status.backends).map(([name, b]) => {
        const isActive = state.status.active === name;
        const loading = state.loading[`backend-${name}`];
        let badge = b.unavailable_reason ? `<span class="status-badge">${b.unavailable_reason}</span>` : '';

        let btn = '';
        if (loading) {
            // This backend is loading - show loading state
            btn = `<button class="btn-primary" disabled>
                <span class="loading"></span>Starting...</button>`;
        } else if (isActive && !anyBackendLoading) {
            // This backend is active and nothing is loading - show stop button
            btn = `<button class="btn-danger" onclick="stopBackend('${name}')">Stop</button>`;
        } else if (b.available && !anyBackendLoading) {
            // This backend is available and nothing is loading - show start/switch button
            btn = `<button class="btn-success" onclick="startBackend('${name}')">
                ${state.status.active ? 'Switch' : 'Start'}</button>`;
        } else if (b.available) {
            // Another backend is loading - disable this button
            btn = `<button class="btn-secondary" disabled>
                ${state.status.active ? 'Switch' : 'Start'}</button>`;
        }

        return `
            <div class="card ${isActive ? 'active' : ''} ${!b.available ? 'unavailable' : ''}">
                <div class="card-header">
                    <div class="card-info">
                        <h3>${b.display_name}${badge}</h3>
                        <div class="subtitle">${b.model_format || 'unknown'}</div>
                    </div>
                    <div>${btn}</div>
                </div>
            </div>
        `;
    }).join('');
}

function renderModels() {
    const section = document.getElementById('models-section');
    const el = document.getElementById('models');

    if (!state.status?.active) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    const activeBackend = state.status.active;
    const allModels = state.models[activeBackend] || [];
    const loadedId = state.backendState?.loaded_model?.id;

    if (allModels.length === 0) {
        el.innerHTML = '<div style="color:#666;font-size:0.8rem;padding:0.5rem;">No models found</div>';
        return;
    }

    // Preserve scroll position
    const scrollTop = el.scrollTop;
    // Check if details was open
    const details = el.querySelector('details.available-models');
    const detailsWasOpen = details?.open;

    // Split into downloaded and available for download
    const downloaded = allModels.filter(m => m.downloaded !== false);
    const available = allModels.filter(m => m.downloaded === false);

    // Sort downloaded: loaded first
    downloaded.sort((a, b) => (b.id === loadedId ? 1 : 0) - (a.id === loadedId ? 1 : 0));

    function getTaskLabel(task) {
        const labels = {
            'chat-completion': 'chat',
            'chat-completions': 'chat',
            'automatic-speech-recognition': 'speech-to-text',
            'embeddings': 'embeddings',
        };
        return labels[task] || task || '';
    }

    function isChatTask(task) {
        return task === 'chat-completion' || task === 'chat-completions' || !task;
    }

    function renderModelCard(m, isAvailable = false) {
        const loading = state.loading[`model-${m.id}`];
        const downloading = state.loading[`download-${m.id}`];
        const isLoaded = m.id === loadedId;
        const isChat = isChatTask(m.task);
        const taskLabel = getTaskLabel(m.task);

        let btn = '';
        if (isLoaded) {
            btn = `<button class="btn-secondary" onclick="unloadModel('${activeBackend}')" ${loading ? 'disabled' : ''}>Unload</button>`;
        } else if (!isAvailable) {
            btn = `<button class="btn-primary" onclick="loadModel('${activeBackend}', '${m.id}')" ${loading ? 'disabled' : ''}>${loading ? '<span class="loading"></span>' : ''}${loadedId ? 'Switch' : 'Load'}</button>`;
        } else {
            btn = `<button class="btn-success" onclick="downloadModel('${activeBackend}', '${m.id}')" ${downloading ? 'disabled' : ''}>${downloading ? '<span class="loading"></span>' : ''}Download</button>`;
        }

        // Show task badge for non-chat models
        const taskBadge = taskLabel && !isChat ? ` <span class="status-badge warning">${taskLabel}</span>` : '';

        return `
            <div class="card ${isLoaded ? 'active' : ''} ${!isChat ? 'non-chat' : ''}" data-model-id="${m.id}">
                <div class="card-header">
                    <div class="card-info">
                        <h3>${m.name}${m.quantization ? ` <span class="status-badge">${m.quantization}</span>` : ''}${taskBadge}${isLoaded ? ' <span class="status-badge loaded">loaded</span>' : ''}</h3>
                        <div class="subtitle">${formatSize(m.size_bytes)}</div>
                    </div>
                    <div>${btn}</div>
                </div>
            </div>
        `;
    }

    let html = downloaded.map(m => renderModelCard(m, false)).join('');

    // Add collapsible section for available models
    if (available.length > 0) {
        html += `
            <details class="available-models" ${detailsWasOpen ? 'open' : ''}>
                <summary>Available for download (${available.length})</summary>
                <div class="available-models-list">
                    ${available.map(m => renderModelCard(m, true)).join('')}
                </div>
            </details>
        `;
    }

    el.innerHTML = html;

    // Restore scroll position
    el.scrollTop = scrollTop;
}

function renderChatHeader() {
    const el = document.getElementById('chat-header');
    const newModelId = state.backendState?.loaded_model?.id || null;

    // Clear chat if model changed
    if (newModelId !== currentModelId) {
        chatMessages = [];
        currentModelId = newModelId;
        renderMessages();
    }

    if (state.backendState?.loaded_model) {
        const model = state.backendState.loaded_model;
        const task = model.task || '';
        const isChat = isChatTask(task);

        if (isChat) {
            el.innerHTML = `<strong>Chat</strong> - ${model.name}`;
        } else {
            const taskLabel = getTaskLabel(task);
            el.innerHTML = `<strong>Chat</strong> - ${model.name} <span style="color:#f59e0b;font-size:0.8rem;">(${taskLabel} model - not for chat)</span>`;
        }
    } else {
        el.innerHTML = '<strong>Chat</strong> - No model loaded';
    }
}

function getTaskLabel(task) {
    const labels = {
        'chat-completion': 'chat',
        'chat-completions': 'chat',
        'automatic-speech-recognition': 'speech-to-text',
        'embeddings': 'embeddings',
    };
    return labels[task] || task || '';
}

function isChatTask(task) {
    return task === 'chat-completion' || task === 'chat-completions' || !task;
}

function isChatModel() {
    const model = state.backendState?.loaded_model;
    if (!model) return false;
    const task = model.task || '';
    return isChatTask(task);
}

function updateChatInput() {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('send-btn');
    const hasModel = !!state.backendState?.loaded_model;
    const canChat = hasModel && isChatModel();

    input.disabled = !canChat || isGenerating;
    btn.disabled = !canChat || isGenerating;

    if (!hasModel) {
        input.placeholder = 'Load a model to start chatting';
    } else if (!isChatModel()) {
        input.placeholder = 'This model does not support chat';
    } else {
        input.placeholder = 'Type a message...';
    }

    const emptyChat = document.getElementById('empty-chat');
    if (emptyChat) {
        emptyChat.style.display = chatMessages.length === 0 ? 'flex' : 'none';
        if (!hasModel) {
            emptyChat.textContent = 'Load a model to start chatting';
        } else if (!isChatModel()) {
            emptyChat.textContent = 'This model does not support chat. Load a chat model instead.';
        } else {
            emptyChat.textContent = 'Start a conversation';
        }
    }
}

function renderMessages() {
    const container = document.getElementById('chat-messages');
    const emptyEl = document.getElementById('empty-chat');

    if (chatMessages.length === 0) {
        container.innerHTML = `<div class="empty-chat" id="empty-chat">${state.backendState?.loaded_model ? 'Start a conversation' : 'Load a model to start chatting'}</div>`;
        return;
    }

    container.innerHTML = chatMessages.map((msg, i) => {
        if (msg.role === 'user') {
            return `<div class="message user">${escapeHtml(msg.content)}</div>`;
        } else {
            let statsHtml = '';
            if (msg.stats && !msg.generating) {
                const s = msg.stats;
                const parts = [];
                if (s.tokens_per_sec) parts.push(`${s.tokens_per_sec.toFixed(1)} tok/s`);
                if (s.time_ms) parts.push(`${(s.time_ms / 1000).toFixed(1)}s`);
                if (s.prompt_tokens && s.completion_tokens) parts.push(`${s.prompt_tokens}→${s.completion_tokens} tokens`);
                if (parts.length) {
                    statsHtml = `<div class="message-stats">${parts.join(' · ')}</div>`;
                }
            }
            const cursor = msg.generating ? '<span class="cursor">▌</span>' : '';
            return `<div class="message assistant"><div class="content">${escapeHtml(msg.content)}${cursor}</div>${statsHtml}</div>`;
        }
    }).join('');

    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text || isGenerating) return;

    input.value = '';
    input.style.height = 'auto';

    chatMessages.push({ role: 'user', content: text });
    chatMessages.push({ role: 'assistant', content: '', generating: true });
    renderMessages();

    isGenerating = true;
    updateChatInput();

    const msgIndex = chatMessages.length - 1;
    abortController = new AbortController();

    try {
        const messages = chatMessages.slice(0, -1).map(m => ({
            role: m.role,
            content: m.content
        }));

        const resp = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages }),
            signal: abortController.signal
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalStats = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = line.slice(6);
                if (data === '[DONE]') continue;

                try {
                    const chunk = JSON.parse(data);

                    if (chunk.choices?.[0]?.delta?.content) {
                        chatMessages[msgIndex].content += chunk.choices[0].delta.content;
                        renderMessages();
                    }

                    // llama.cpp sends timings in final chunk
                    if (chunk.timings) {
                        finalStats = finalStats || {};
                        finalStats.tokens_per_sec = chunk.timings.predicted_per_second;
                        finalStats.time_ms = chunk.timings.predicted_ms;
                        finalStats.prompt_tokens = chunk.timings.prompt_n;
                        finalStats.completion_tokens = chunk.timings.predicted_n;
                    }

                    // OpenAI-style usage (fallback)
                    if (chunk.usage) {
                        finalStats = finalStats || {};
                        finalStats.prompt_tokens = finalStats.prompt_tokens || chunk.usage.prompt_tokens;
                        finalStats.completion_tokens = finalStats.completion_tokens || chunk.usage.completion_tokens;
                    }
                } catch (e) {}
            }
        }

        chatMessages[msgIndex].generating = false;
        if (finalStats) {
            chatMessages[msgIndex].stats = finalStats;
        }
        renderMessages();

    } catch (e) {
        if (e.name !== 'AbortError') {
            chatMessages[msgIndex].content = `Error: ${e.message}`;
            chatMessages[msgIndex].generating = false;
        }
        renderMessages();
    } finally {
        isGenerating = false;
        abortController = null;
        updateChatInput();
    }
}

// Event listeners
document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

document.getElementById('chat-input').addEventListener('input', (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
});

document.getElementById('send-btn').addEventListener('click', sendMessage);

// Fire-and-forget API calls - loading states and toasts come from SSE events
async function startBackend(name) {
    try {
        const resp = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend: name })
        });
        const result = await resp.json();
        // Only show error if API itself failed (validation error)
        if (!result.success) {
            showToast(result.message, 'error');
        }
        // Success toast and loading state come from SSE
    } catch (e) {
        showToast('Failed to start backend', 'error');
    }
}

async function stopBackend(name) {
    try {
        const resp = await fetch(`/api/stop/${name}`, { method: 'POST' });
        const result = await resp.json();
        if (!result.success) {
            showToast(result.message, 'error');
        }
    } catch (e) {
        showToast('Failed to stop backend', 'error');
    }
}

async function loadModel(backend, modelId) {
    try {
        const resp = await fetch('/api/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend, model_id: modelId })
        });
        const result = await resp.json();
        if (!result.success) {
            showToast(result.message, 'error');
        }
    } catch (e) {
        showToast('Failed to load model', 'error');
    }
}

async function unloadModel(backend) {
    try {
        const resp = await fetch(`/api/models/unload/${backend}`, { method: 'POST' });
        const result = await resp.json();
        if (!result.success) {
            showToast(result.message, 'error');
        }
    } catch (e) {
        showToast('Failed to unload model', 'error');
    }
}

async function downloadModel(backend, modelId) {
    state.loading[`download-${modelId}`] = true;
    render();
    try {
        const resp = await fetch('/api/models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend, model_id: modelId })
        });
        const result = await resp.json();
        showToast(result.message, result.success ? 'success' : 'error');
        if (result.success) {
            await fetchAll();
        }
    } finally {
        state.loading[`download-${modelId}`] = false;
        render();
    }
}

function showToast(message, type) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// Initialize: connect SSE and fetch initial models
connectSSE();
fetchModels();
