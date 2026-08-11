const mapContainer = document.getElementById('world-map');
const searchInput = document.getElementById('course-search');
const searchResults = document.getElementById('search-results');
const featureCountEl = document.getElementById('feature-count');

const MAX_RESULTS = 25;

function normalize(value) {
  return String(value || '').toLowerCase().trim();
}

function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function buildPrefixIndex(features) {
  const index = new Map();

  features.forEach((feature, idx) => {
    const name = normalize(feature.properties?.name);
    const region = normalize(feature.properties?.province);
    const combined = `${name} ${region}`;
    feature.properties._search = combined;

    const tokens = new Set(combined.split(/\s+/).filter(Boolean));
    for (const token of tokens) {
      const key3 = token.slice(0, 3);
      if (!key3) {
        continue;
      }
      if (!index.has(key3)) {
        index.set(key3, []);
      }
      index.get(key3).push(idx);
    }
  });

  return index;
}

function pickSearchKey(query) {
  const tokens = query.split(/\s+/).filter((token) => token.length >= 3);
  if (!tokens.length) {
    return query.slice(0, 3);
  }

  tokens.sort((left, right) => right.length - left.length);
  return tokens[0].slice(0, 3);
}

function renderResults(results, onSelect) {
  searchResults.innerHTML = '';

  if (!results.length) {
    return;
  }

  for (const feature of results) {
    const li = document.createElement('li');
    const button = document.createElement('button');
    const courseName = feature.properties?.name || 'Unnamed Course';
    const province = feature.properties?.province || 'Unknown Region';
    button.innerHTML = `${courseName}<span class="search-sub">${province}</span>`;
    button.addEventListener('click', () => onSelect(feature));
    li.appendChild(button);
    searchResults.appendChild(li);
  }
}

function createPopupHTML(feature) {
  const props = feature.properties || {};
  const name = props.name || 'Unnamed Course';
  const province = props.province || 'Unknown Region';
  const id = props.id || 'Unknown';

  return `
    <strong>${name}</strong><br>
    <span>${province}</span><br>
    <small>ID: ${id}</small>
  `;
}

function getInitialQuery() {
  const params = new URLSearchParams(window.location.search);
  return normalize(params.get('q') || '');
}

async function loadCourses() {
  const response = await fetch('data/golf_courses.geojson', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`GeoJSON fetch failed (${response.status})`);
  }
  return response.json();
}

async function init() {
  const geojson = await loadCourses();
  const features = geojson.features || [];
  const allIndexes = features.map((_, idx) => idx);

  if (featureCountEl) {
    featureCountEl.textContent = `${new Intl.NumberFormat('en-US').format(features.length)} courses loaded`;
  }

  const prefixIndex = buildPrefixIndex(features);

  const map = new maplibregl.Map({
    container: mapContainer,
    style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    center: [-15, 32],
    zoom: 2,
    minZoom: 1.3,
    attributionControl: true,
    maxPitch: 0,
  });

  map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'top-right');

  map.on('load', () => {
    map.addSource('courses', {
      type: 'geojson',
      data: geojson,
      cluster: true,
      clusterRadius: 52,
      clusterMaxZoom: 9,
      generateId: true,
    });

    map.addLayer({
      id: 'clusters',
      type: 'circle',
      source: 'courses',
      filter: ['has', 'point_count'],
      paint: {
        'circle-color': [
          'step',
          ['get', 'point_count'],
          '#5d8e90',
          60,
          '#e58a3a',
          300,
          '#b54e15',
        ],
        'circle-radius': [
          'step',
          ['get', 'point_count'],
          16,
          60,
          23,
          300,
          30,
        ],
        'circle-stroke-color': '#fff',
        'circle-stroke-width': 1.25,
      },
    });

    map.addLayer({
      id: 'cluster-count',
      type: 'symbol',
      source: 'courses',
      filter: ['has', 'point_count'],
      layout: {
        'text-field': '{point_count_abbreviated}',
        'text-font': ['Noto Sans Bold'],
        'text-size': 12,
      },
      paint: {
        'text-color': '#fcfcfa',
      },
    });

    map.addLayer({
      id: 'unclustered-point',
      type: 'circle',
      source: 'courses',
      filter: ['!', ['has', 'point_count']],
      paint: {
        'circle-color': '#1e595f',
        'circle-radius': 4,
        'circle-stroke-width': 1,
        'circle-stroke-color': '#f4f4ef',
      },
    });

    map.on('click', 'clusters', (event) => {
      const feature = map.queryRenderedFeatures(event.point, {
        layers: ['clusters'],
      })[0];
      if (!feature) {
        return;
      }
      const clusterId = feature.properties.cluster_id;
      map.getSource('courses').getClusterExpansionZoom(clusterId, (error, zoom) => {
        if (error) {
          return;
        }
        map.easeTo({
          center: feature.geometry.coordinates,
          zoom,
          duration: 400,
        });
      });
    });

    map.on('click', 'unclustered-point', (event) => {
      const feature = event.features?.[0];
      if (!feature) {
        return;
      }
      new maplibregl.Popup({ closeButton: true })
        .setLngLat(feature.geometry.coordinates)
        .setHTML(createPopupHTML(feature))
        .addTo(map);
    });

    map.on('mouseenter', 'clusters', () => {
      map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', 'clusters', () => {
      map.getCanvas().style.cursor = '';
    });
    map.on('mouseenter', 'unclustered-point', () => {
      map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', 'unclustered-point', () => {
      map.getCanvas().style.cursor = '';
    });

    const search = debounce(() => {
      const query = normalize(searchInput.value);
      if (query.length < 2) {
        renderResults([], null);
        return;
      }

      const key = pickSearchKey(query);
      const candidateIndexes = prefixIndex.get(key) || allIndexes;
      const results = [];

      for (const idx of candidateIndexes) {
        const feature = features[idx];
        if (!feature?.properties?._search?.includes(query)) {
          continue;
        }
        results.push(feature);
        if (results.length >= MAX_RESULTS) {
          break;
        }
      }

      renderResults(results, (feature) => {
        const [lon, lat] = feature.geometry.coordinates;
        map.flyTo({ center: [lon, lat], zoom: 11, speed: 0.8 });
        new maplibregl.Popup({ closeButton: true })
          .setLngLat([lon, lat])
          .setHTML(createPopupHTML(feature))
          .addTo(map);
      });
    }, 130);

    searchInput.addEventListener('input', search);

    const initialQuery = getInitialQuery();
    if (initialQuery && searchInput) {
      searchInput.value = initialQuery;
      search();
    }
  });
}

init().catch((error) => {
  if (featureCountEl) {
    featureCountEl.textContent = 'Failed to load map dataset. Run scripts/build_site.py first.';
  }
  console.error(error);
});
