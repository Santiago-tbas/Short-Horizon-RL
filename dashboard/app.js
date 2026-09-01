document.addEventListener('DOMContentLoaded', () => {
    // API endpoint definitions
    const API_URL = 'http://localhost:8000';
    
    // State storage
    let projectData = {
        environments: [],
        results_mab: [],
        screenshots: []
    };

    // DOM Elements
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const tabTitle = document.getElementById('tab-title');
    const tabDescription = document.getElementById('tab-description');
    const refreshBtn = document.getElementById('refresh-btn');

    // Tab Switching Logic
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            
            // Toggle active nav class
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Toggle active tab pane
            tabPanes.forEach(pane => {
                if (pane.id === `tab-${targetTab}`) {
                    pane.classList.add('active');
                } else {
                    pane.classList.remove('active');
                }
            });

            // Update Header Text
            updateHeader(targetTab);
        });
    });

    function updateHeader(tab) {
        switch(tab) {
            case 'overview':
                tabTitle.innerText = "Project Overview";
                tabDescription.innerText = "Registered environments, architecture and setup details.";
                break;
            case 'epsilon-sweep':
                tabTitle.innerText = "Epsilon Precision Boundary";
                tabDescription.innerText = "Interactive sweep-line boundary simulation over sub-optimality gaps.";
                break;
            case 'regret-plots':
                tabTitle.innerText = "MAB Regret Analysis";
                tabDescription.innerText = "Cumulative regret comparison generated from experiments.";
                break;
            case 'gridworld-walk':
                tabTitle.innerText = "Gridworld Simulation Steps";
                tabDescription.innerText = "Visualizing the trajectory of a trained agent step-by-step.";
                break;
            case 'experiment-runner':
                tabTitle.innerText = "Live Experiment Runner";
                tabDescription.innerText = "Trigger reinforcement learning scripts and view execution output.";
                break;
        }
    }

    // ----------------------------------------------------
    // API DATA SYNC
    // ----------------------------------------------------
    async function syncProjectData() {
        refreshBtn.disabled = true;
        refreshBtn.innerText = "🔄 Syncing...";

        try {
            // Fetch environments
            const envResponse = await fetch(`${API_URL}/api/environments`);
            const envData = await envResponse.json();
            if (envData.status === 'success') {
                projectData.environments = envData.environments;
            }

            // Fetch results and screenshots
            const resResponse = await fetch(`${API_URL}/api/results`);
            const resData = await resResponse.json();
            if (resData.status === 'success') {
                projectData.results_mab = resData.results_mab;
                projectData.screenshots = resData.screenshots;
            }

            // Update UI elements
            renderEnvironments();
            renderPlotsMenu();
            initGridworldTrajectory();

        } catch (error) {
            console.error("Error syncing project data:", error);
        } finally {
            refreshBtn.disabled = false;
            refreshBtn.innerText = "🔄 Sync Project Data";
        }
    }

    refreshBtn.addEventListener('click', syncProjectData);

    // ----------------------------------------------------
    // OVERVIEW TAB: RENDER ENVIRONMENTS
    // ----------------------------------------------------
    function renderEnvironments() {
        const container = document.getElementById('envs-container');
        if (projectData.environments.length === 0) {
            container.innerHTML = `<li class="loading">No registered environments found.</li>`;
            return;
        }

        container.innerHTML = projectData.environments.map(env => `
            <li>🎮 ${env}</li>
        `).join('');
    }

    // ----------------------------------------------------
    // EPSILON SWEEP INTERACTIVE SIMULATION
    // ----------------------------------------------------
    // Static sub-optimality gaps matching epsilon.py
    const banditGaps = [
        0.0, 0.02, 0.04, 0.07, 0.10, 0.14, 0.18,
        0.24, 0.32, 0.42, 0.55, 0.70, 0.88, 1.05, 1.30
    ];

    // DOM Controls
    const sliderT = document.getElementById('param-t');
    const sliderSigma = document.getElementById('param-sigma');
    const sliderEta = document.getElementById('param-eta');
    
    const valT = document.getElementById('val-t');
    const valSigma = document.getElementById('val-sigma');
    const valEta = document.getElementById('val-eta');

    const statCt = document.getElementById('stat-ct');
    const statEps = document.getElementById('stat-eps');

    const vizBand = document.getElementById('viz-band-element');
    const vizBoundary = document.getElementById('viz-boundary-line');
    const vizGapsLayer = document.getElementById('viz-gaps-layer');

    function computeEpsilonGaussian(gaps, T, sigma, eta) {
        // c_T = sqrt(2 * sigma^2 * log(T) / (eta * T))
        const c_T = Math.sqrt((2 * Math.pow(sigma, 2) * Math.log(T)) / (eta * T));
        const sortedGaps = [...new Set(gaps)].sort((a, b) => a - b);
        
        let epsilon = 0.0;
        for (let i = 0; i < sortedGaps.length - 1; i++) {
            if (sortedGaps[i+1] - sortedGaps[i] <= c_T) {
                epsilon = sortedGaps[i+1];
            } else {
                break;
            }
        }
        return { epsilon, c_T };
    }

    function updateSimulation() {
        const T = parseInt(sliderT.value);
        const sigma = parseFloat(sliderSigma.value);
        const eta = parseFloat(sliderEta.value);

        // Update control labels
        valT.innerText = T;
        valSigma.innerText = sigma.toFixed(1);
        valEta.innerText = eta.toFixed(2);

        // Calculate
        const { epsilon, c_T } = computeEpsilonGaussian(banditGaps, T, sigma, eta);

        // Update stats header
        statCt.innerText = c_T.toFixed(4);
        statEps.innerText = epsilon.toFixed(4);

        // Render visualization elements
        // The axis is bounded between -0.05 and max(gaps) + 0.12 (approx -0.05 to 1.45)
        const minAxis = -0.05;
        const maxAxis = 1.42;
        const axisRange = maxAxis - minAxis;

        // Position converter: gap value to percentage width (5% margin left/right)
        const getPct = (val) => {
            const ratio = (val - minAxis) / axisRange;
            return 5 + ratio * 90; // scale to 5%-95%
        };

        // Update red band position (width and left offset)
        const bandStartPct = getPct(epsilon);
        const bandEndPct = getPct(epsilon + c_T);
        
        vizBand.style.left = `${bandStartPct}%`;
        vizBand.style.width = `${bandEndPct - bandStartPct}%`;

        // Update boundary line position
        vizBoundary.style.left = `${bandStartPct}%`;

        // Render dots representing bandit gaps
        vizGapsLayer.innerHTML = banditGaps.map((gap, index) => {
            let colorClass = 'blue';
            if (gap <= epsilon + 1e-9) {
                colorClass = 'green';
            } else if (gap > epsilon && gap <= epsilon + c_T + 1e-9) {
                colorClass = 'red';
            }

            // Alternating label text heights to prevent overlap (3 tiers)
            const heightTiers = ['25%', '35%', '45%', '55%', '65%', '75%'];
            const height = heightTiers[index % 3];
            const direction = (index % 2 === 0) ? '1' : '-1';
            
            // Dynamic position on axis (vertical midpoint is 50%)
            const topOffset = direction === '1' ? (50 - (index % 3) * 12) : (50 + (index % 3) * 12);

            const leftPct = getPct(gap);

            return `
                <div class="gap-dot ${colorClass}" style="left: ${leftPct}%; top: 50%;"></div>
                <div class="gap-label ${colorClass}" style="left: ${leftPct}%; top: ${topOffset}%;">
                    Δ=${gap.toFixed(2)}
                </div>
            `;
        }).join('');
    }

    // Attach listeners to sliders
    [sliderT, sliderSigma, sliderEta].forEach(slider => {
        slider.addEventListener('input', updateSimulation);
    });

    // Run initial simulation calculation
    updateSimulation();


    // ----------------------------------------------------
    // REGRET PLOTS BROWSER
    // ----------------------------------------------------
    function renderPlotsMenu() {
        const plotsList = document.getElementById('plots-list');
        const viewPort = document.getElementById('plot-viewport');
        const detailsPanel = document.getElementById('plot-details');

        // Filter MAB result files
        const plotFiles = projectData.results_mab.filter(f => f.endsWith('.png'));

        if (plotFiles.length === 0) {
            plotsList.innerHTML = `<li class="loading">No plots found. Run experiments first.</li>`;
            return;
        }

        plotsList.innerHTML = plotFiles.map(file => `
            <li class="plot-item" data-filename="${file}">${file}</li>
        `).join('');

        // Attach click handlers
        const plotItems = document.querySelectorAll('.plot-item');
        plotItems.forEach(item => {
            item.addEventListener('click', () => {
                plotItems.forEach(pi => pi.classList.remove('active'));
                item.classList.add('active');

                const filename = item.getAttribute('data-filename');
                const imageSrc = `${API_URL}/api/image/results_mab/${filename}`;

                // Render image
                viewPort.innerHTML = `<img src="${imageSrc}" alt="${filename}">`;

                // Update details (parse timestamp/metadata if possible)
                const logMatch = filename.match(/_(\d+\.\d+)(_ylog)?\.png$/);
                let timestamp = "Unknown";
                if (logMatch && logMatch[1]) {
                    const dateObj = new Date(parseFloat(logMatch[1]) * 1000);
                    timestamp = dateObj.toLocaleString();
                }

                detailsPanel.innerHTML = `
                    <h3>Plot Metadata</h3>
                    <p><strong>Filename:</strong> ${filename}</p>
                    <p><strong>Experiment Date:</strong> ${timestamp}</p>
                    <p><strong>Path:</strong> <code>results_mab/${filename}</code></p>
                `;
            });
        });

        // Select first item by default
        if (plotItems.length > 0) {
            plotItems[0].click();
        }
    }


    // ----------------------------------------------------
    // GRIDWORLD WALKTHROUGH TRAJECTORY SLIDESHOW
    // ----------------------------------------------------
    let gwFrames = [];
    let gwCurrentIndex = 0;
    let gwPlayInterval = null;

    const trajPrev = document.getElementById('traj-prev');
    const trajNext = document.getElementById('traj-next');
    const trajPlay = document.getElementById('traj-play');
    const trajStepNum = document.getElementById('traj-step-num');
    const trajTotalSteps = document.getElementById('traj-total-steps');
    const trajFrameImg = document.getElementById('trajectory-frame-img');
    const trajOverlay = document.getElementById('trajectory-loading-overlay');

    function initGridworldTrajectory() {
        // Group screenshots by gridworld timestamp
        // screenshots are like: Gridworld-1779560839.9038491-0.png, Gridworld-1779560839.9038491-1.png...
        const frameFiles = projectData.screenshots.filter(f => f.startsWith('Gridworld-') && f.includes('-'));
        
        if (frameFiles.length === 0) {
            trajStepNum.innerText = "0";
            trajTotalSteps.innerText = "0";
            trajFrameImg.src = "";
            trajFrameImg.alt = "No Gridworld frames available yet.";
            return;
        }

        // Parse unique experiment run IDs (the float timestamp)
        const runIds = [...new Set(frameFiles.map(f => {
            const match = f.match(/Gridworld-(\d+\.\d+)-\d+\.png/);
            return match ? match[1] : null;
        }).filter(Boolean))];

        if (runIds.length === 0) return;

        // Select the latest run ID (most recent experiment)
        const latestRunId = runIds.sort((a, b) => parseFloat(b) - parseFloat(a))[0];

        // Filter and sort frames for the latest run
        gwFrames = frameFiles.filter(f => f.includes(latestRunId)).sort((a, b) => {
            const stepA = parseInt(a.match(/Gridworld-\d+\.\d+-(\d+)\.png/)[1]);
            const stepB = parseInt(b.match(/Gridworld-\d+\.\d+-(\d+)\.png/)[1]);
            return stepA - stepB;
        });

        gwCurrentIndex = 0;
        trajTotalSteps.innerText = gwFrames.length;
        
        renderGridworldFrame();
    }

    function renderGridworldFrame() {
        if (gwFrames.length === 0) return;

        const filename = gwFrames[gwCurrentIndex];
        const stepNum = gwCurrentIndex;
        
        trajStepNum.innerText = stepNum;
        trajOverlay.style.display = 'block';

        // Load image
        const imgUrl = `${API_URL}/api/image/screenshots/${filename}`;
        
        const tempImg = new Image();
        tempImg.onload = () => {
            trajFrameImg.src = imgUrl;
            trajOverlay.style.display = 'none';
        };
        tempImg.src = imgUrl;
    }

    trajPrev.addEventListener('click', () => {
        if (gwFrames.length === 0) return;
        stopGwAutoplay();
        gwCurrentIndex = (gwCurrentIndex - 1 + gwFrames.length) % gwFrames.length;
        renderGridworldFrame();
    });

    trajNext.addEventListener('click', () => {
        if (gwFrames.length === 0) return;
        stopGwAutoplay();
        gwCurrentIndex = (gwCurrentIndex + 1) % gwFrames.length;
        renderGridworldFrame();
    });

    function stopGwAutoplay() {
        if (gwPlayInterval) {
            clearInterval(gwPlayInterval);
            gwPlayInterval = null;
            trajPlay.innerText = "Play Auto";
            trajPlay.classList.remove('btn-primary');
            trajPlay.classList.add('btn-secondary');
        }
    }

    trajPlay.addEventListener('click', () => {
        if (gwFrames.length === 0) return;
        if (gwPlayInterval) {
            stopGwAutoplay();
        } else {
            trajPlay.innerText = "Pause";
            trajPlay.classList.remove('btn-secondary');
            trajPlay.classList.add('btn-primary');
            
            gwPlayInterval = setInterval(() => {
                gwCurrentIndex = (gwCurrentIndex + 1) % gwFrames.length;
                renderGridworldFrame();
            }, 600); // cycle frames every 600ms
        }
    });


    // ----------------------------------------------------
    // LIVE EXPERIMENT RUNNER
    // ----------------------------------------------------
    let selectedScript = null;
    const expOptions = document.querySelectorAll('.exp-runner-option');
    const runBtn = document.getElementById('run-exp-btn');
    const terminalOutput = document.getElementById('terminal-output');
    const terminalStatus = document.getElementById('terminal-status');

    expOptions.forEach(opt => {
        opt.addEventListener('click', () => {
            expOptions.forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            
            selectedScript = opt.getAttribute('data-script');
            runBtn.disabled = false;
        });
    });

    runBtn.addEventListener('click', async () => {
        if (!selectedScript) return;

        // UI state: running
        runBtn.disabled = true;
        terminalStatus.innerText = "Running...";
        terminalStatus.className = "badge badge-accent";
        terminalOutput.innerText = `>>> Launching Python script: ${selectedScript} via venv...\n>>> Running simulations, please wait. This might take 10-20 seconds...\n`;

        try {
            const response = await fetch(`${API_URL}/api/run_experiment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script: selectedScript })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                terminalStatus.innerText = data.returncode === 0 ? "Success" : "Error";
                terminalStatus.className = data.returncode === 0 ? "badge badge-primary" : "badge badge-accent";
                
                let output = "";
                if (data.stdout) output += `STDOUT:\n${data.stdout}\n`;
                if (data.stderr) output += `STDERR:\n${data.stderr}\n`;
                
                terminalOutput.innerText = output || "Execution completed with no console output.";
                
                // Automatically sync data to capture new logs/plots
                syncProjectData();

            } else {
                terminalStatus.innerText = "Failed";
                terminalStatus.className = "badge badge-accent";
                terminalOutput.innerText = `Error launching experiment: ${data.message}`;
            }

        } catch (error) {
            terminalStatus.innerText = "Network Error";
            terminalStatus.className = "badge badge-accent";
            terminalOutput.innerText = `Connection lost: ${error.message}`;
        } finally {
            runBtn.disabled = false;
        }
    });


    // Initial setup sync
    syncProjectData();
});
