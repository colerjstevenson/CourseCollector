const metricEls = {
  totalCourses: document.querySelector('[data-metric="totalCourses"]'),
  sourceRegions: document.querySelector('[data-metric="sourceRegions"]'),
  builtAt: document.querySelector('[data-metric="builtAt"]'),
  totalCities: document.querySelector('[data-metric="totalCities"]'),
};

const coveragePills = document.getElementById('coverage-pills');
const visualLinks = document.getElementById('visual-links');
const dataDownloadLinks = document.getElementById('data-download-links');
const citySearchInput = document.getElementById('city-search');
const cityCards = document.getElementById('city-cards');
const MAX_CITY_RENDER = 250;

function formatNumber(value) {
  return new Intl.NumberFormat('en-US').format(value);
}

function addLinks(targetEl, links) {
  if (!targetEl || !Array.isArray(links)) {
    return;
  }

  targetEl.innerHTML = '';

  if (!links.length) {
    const empty = document.createElement('li');
    empty.textContent = 'No links configured yet.';
    targetEl.appendChild(empty);
    return;
  }

  for (const link of links) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = link.href;
    a.textContent = link.label;
    if (link.external) {
      a.target = '_blank';
      a.rel = 'noreferrer noopener';
    }
    li.appendChild(a);
    targetEl.appendChild(li);
  }
}

function normalize(value) {
  return String(value || '').toLowerCase().trim();
}

function formatCurrency(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A';
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value);
}

async function loadSummary() {
  const response = await fetch('data/landing_summary.json', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Summary fetch failed (${response.status})`);
  }
  return response.json();
}

async function loadLinks() {
  const response = await fetch('data/site_links.json', { cache: 'no-store' });
  if (!response.ok) {
    return { visuals: [], dataDownloads: [] };
  }
  return response.json();
}

async function loadCities() {
  const response = await fetch('data/cities.json', { cache: 'no-store' });
  if (!response.ok) {
    return [];
  }
  return response.json();
}

function renderCoverage(items) {
  if (!coveragePills) {
    return;
  }

  coveragePills.innerHTML = '';
  for (const item of items) {
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.textContent = `${item.name}: ${formatNumber(item.count)}`;
    coveragePills.appendChild(pill);
  }
}

function cityCardHTML(city) {
  const province = city.province || city.country || 'Unknown region';
  const population = typeof city.population === 'number' ? formatNumber(city.population) : 'N/A';
  const courses = formatNumber(city.golf_course_count || 0);
  const amenities = formatNumber(city.amenity_total_count || 0);
  const income = formatCurrency(city.median_household_income);
  const topAmenity = city.top_amenity || 'N/A';
  const worldMapQuery = encodeURIComponent(city.city_name || city.city_slug || '');

  return `
    <article class="city-card">
      <div class="city-card-head">
        <h3>${city.city_name || city.city_slug}</h3>
        <span>${province}</span>
      </div>
      <p class="city-meta">Population: ${population}</p>
      <p class="city-meta">Golf courses: ${courses}</p>
      <p class="city-meta">Amenity points: ${amenities}</p>
      <p class="city-meta">Median household income: ${income}</p>
      <p class="city-meta">Top amenity: ${topAmenity}</p>
      <a class="city-link" href="world-map.html?q=${worldMapQuery}">Open on world map</a>
    </article>
  `;
}

function renderCities(cities) {
  if (!cityCards) {
    return;
  }

  cityCards.innerHTML = '';
  if (!cities.length) {
    cityCards.innerHTML = '<p class="empty-city-state">No city records available for this build.</p>';
    return;
  }

  const visible = cities.slice(0, MAX_CITY_RENDER);
  const limited = cities.length > visible.length;
  const cardsHtml = visible.map(cityCardHTML).join('');
  const notice = limited
    ? `<p class="empty-city-state">Showing first ${MAX_CITY_RENDER} of ${cities.length} cities. Refine your search for a narrower list.</p>`
    : '';

  cityCards.innerHTML = `${notice}${cardsHtml}`;
}

async function init() {
  try {
    const [summary, links, cities] = await Promise.all([loadSummary(), loadLinks(), loadCities()]);

    if (metricEls.totalCourses) {
      metricEls.totalCourses.textContent = formatNumber(summary.total_courses || 0);
    }
    if (metricEls.sourceRegions) {
      metricEls.sourceRegions.textContent = formatNumber(summary.source_region_count || 0);
    }
    if (metricEls.builtAt) {
      metricEls.builtAt.textContent = summary.generated_at_utc || 'Unknown';
    }
    if (metricEls.totalCities) {
      metricEls.totalCities.textContent = formatNumber(summary.total_cities || cities.length || 0);
    }

    renderCoverage(summary.top_source_regions || []);
    addLinks(visualLinks, links.visuals);
    addLinks(dataDownloadLinks, links.dataDownloads);
    renderCities(cities);

    if (citySearchInput) {
      citySearchInput.addEventListener('input', () => {
        const query = normalize(citySearchInput.value);
        if (!query) {
          renderCities(cities);
          return;
        }

        const filtered = cities.filter((city) => {
          const haystack = [city.city_name, city.city_slug, city.province, city.country]
            .map(normalize)
            .join(' ');
          return haystack.includes(query);
        });
        renderCities(filtered);
      });
    }
  } catch (error) {
    if (metricEls.totalCourses) {
      metricEls.totalCourses.textContent = 'Build data missing';
    }
    if (coveragePills) {
      coveragePills.textContent = 'Run scripts/build_site.py to generate landing data artifacts.';
    }
    if (cityCards) {
      cityCards.innerHTML = '<p class="empty-city-state">City data unavailable.</p>';
    }
    console.error(error);
  }
}

init();
