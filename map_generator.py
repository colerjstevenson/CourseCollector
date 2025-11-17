"""
Golf Course Map Generator for Canada

This module reads golf course data from a CSV file and generates an interactive
map of Canada with pins at each unique latitude/longitude location.
"""

import pandas as pd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time
import os
from urllib.parse import urlparse
from datetime import datetime


# Global variables for API handler
csv_path_global = None


class CustomRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler to support API endpoints and CORS."""
    
    def do_POST(self):
        """Handle POST requests for API endpoints."""
        if self.path == '/api/update_row':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                row_idx = data.get('rowIdx')
                updates = data.get('updates', {})
                source_file = data.get('source_file')
                source_index = data.get('source_index')

                # Determine target CSV and row index
                if source_file:
                    target_csv = os.path.abspath(source_file)
                    target_row = int(source_index)
                else:
                    # fallback to global csv and provided row_idx
                    if isinstance(csv_path_global, (list, tuple)):
                        target_csv = csv_path_global[0]
                    else:
                        target_csv = csv_path_global
                    target_row = int(row_idx)

                # Save using helper to keep behavior consistent
                save_row_data(target_csv, target_row, updates)
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True}).encode('utf-8'))
                print(f"Row {row_idx} updated successfully")
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def load_golf_courses(csv_path: str) -> pd.DataFrame:
    """
    Load golf course data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing golf course data
        
    Returns:
        DataFrame with golf course data
    """
    df = pd.read_csv(csv_path)
    # Ensure there's a column to track manual edits
    if 'manually_edited' not in df.columns:
        df['manually_edited'] = False
    return df


def load_multiple_csvs(csv_paths):
    """Load multiple CSV files and annotate rows with source info.

    Args:
        csv_paths: list of CSV file paths

    Returns:
        Combined DataFrame with additional columns: _source_file, _source_index
    """
    repo_root = Path(__file__).parent.resolve()
    frames = []
    for path in csv_paths:
        p = Path(path)
        df = pd.read_csv(p)
        # Ensure tracking column exists
        if 'manually_edited' not in df.columns:
            df['manually_edited'] = False

        # Compute relative path (relative to repo root)
        try:
            rel = str(p.resolve().relative_to(repo_root))
        except Exception:
            rel = os.path.relpath(str(p.resolve()), str(repo_root))

        # Annotate source file and original row index
        df['_source_file'] = rel
        df['_source_index'] = df.index.astype(int)

        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined


def save_row_data(csv_path: str, row_index: int, updated_data: dict):
    """
    Save updated row data back to the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        row_index: Index of the row to update
        updated_data: Dictionary of updated column values
    """
    df = pd.read_csv(csv_path)
    # Add column if missing
    if 'manually_edited' not in df.columns:
        df['manually_edited'] = False

    for col, value in updated_data.items():
        if col in df.columns:
            df.at[row_index, col] = value if value != '' else None

    # Mark as manually edited with UTC ISO timestamp
    df.at[row_index, 'manually_edited'] = datetime.utcnow().isoformat()

    df.to_csv(csv_path, index=False)
    print(f"Row {row_index} updated and saved to {csv_path}")


def get_unique_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique latitude/longitude pairs from the dataframe.
    
    Args:
        df: DataFrame containing golf course data
        
    Returns:
        DataFrame with unique locations and associated course info
    """
    # Remove rows with missing latitude or longitude
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    
    # Get unique locations
    unique_locs = df_clean.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
    
    print(f"Total records: {len(df)}")
    print(f"Records with valid coordinates: {len(df_clean)}")
    print(f"Unique locations: {len(unique_locs)}")
    
    return unique_locs


def create_canada_map(df: pd.DataFrame, unique_locations: pd.DataFrame, csv_path: str = None, output_file: str = "golf_courses_map.html") -> folium.Map:
    """
    Create an interactive map of Canada with golf course pins.
    
    Args:
        df: Full DataFrame with all data (needed for row lookups)
        unique_locations: DataFrame with unique golf course locations
        csv_path: Path to the original CSV file
        output_file: Name of the output HTML file
        
    Returns:
        Folium map object
    """
    # Calculate center of map (rough center of Canada)
    center_lat = unique_locations['latitude'].mean()
    center_lon = unique_locations['longitude'].mean()
    
    print(f"Map center: ({center_lat:.4f}, {center_lon:.4f})")
    
    # Create base map
    golf_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add marker cluster for better visualization at various zoom levels
    marker_cluster = MarkerCluster().add_to(golf_map)
    
    # Add individual markers
    for idx, row in unique_locations.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        # Build popup text with all available information from the row
        popup_parts = []
        
        # Add all columns from the row as key-value pairs (skip internal source keys)
        for col, value in row.items():
            # Skip latitude/longitude as we already display them
            if col in ['latitude', 'longitude']:
                continue
            # Skip internal keys that start with underscore
            if isinstance(col, str) and col.startswith('_'):
                continue

            # Format the value, handling NaN and None
            if pd.isna(value):
                display_value = '???'
            else:
                value_str = str(value)
                # Check if value is a URL and convert to clickable link
                if ("www" in value_str and ".com" in value_str) or ("http" in value_str) or ('.ca' in value_str):
                    display_value = f'<a href="{value_str}" target="_blank">{value_str}</a>'
                else:
                    display_value = value_str

            # Create a readable label from column name
            label = col.replace('_', ' ').title()
            popup_parts.append(f"<b>{label}:</b> {display_value}")
        
        # Create HTML for edit button with embedded JavaScript
        gcid = row.get('gcid', 'unknown')
        edit_button = f'''
        <br><br>
        <button onclick="editRow({idx}, '{gcid}')" style="background-color:#4CAF50;color:white;padding:10px;border:none;border-radius:4px;cursor:pointer;">Edit Row</button>
        '''

        popup_text = "<br>".join(popup_parts) + edit_button
        
        # Create marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=400),
            tooltip=row.get('CourseName', 'Golf Course'),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)
    
    # Build the rowDataMap from unique_locations (keys are unique_locations indexes)
    row_data_map = {str(idx): row.to_dict() for idx, row in unique_locations.iterrows()}

    # Make sure source paths in row_data_map are relative (already set in combined df)
    # Inject custom JavaScript for edit functionality
    edit_js = '''
    <script>
    function editRow(rowIdx, gcid) {
        const rowData = window.rowDataMap[rowIdx];
        let formHtml = '<div style="max-height: 500px; overflow-y: auto;">';
        
        for (const [key, value] of Object.entries(rowData)) {
            // Skip internal source keys
            if (typeof key === 'string' && key.startsWith('_')) continue;

            const displayValue = value === null || value === undefined ? '' : value;

            // Show manually_edited as read-only
            if (key === 'manually_edited') {
                formHtml += `<div style="margin-bottom: 10px;">
                    <label style="font-weight: bold; display: block; margin-bottom: 5px;">${key}:</label>
                    <input type="text" id="edit_${key}" value="${displayValue}" style="width: 100%; padding: 5px; border: 1px solid #ccc; border-radius: 3px;" readonly>
                </div>`;
            } else {
                formHtml += `<div style="margin-bottom: 10px;">
                    <label style="font-weight: bold; display: block; margin-bottom: 5px;">${key}:</label>
                    <input type="text" id="edit_${key}" value="${displayValue}" style="width: 100%; padding: 5px; border: 1px solid #ccc; border-radius: 3px;">
                </div>`;
            }
        }
        
        formHtml += '</div>';
        
        const modal = document.createElement('div');
        modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center; z-index: 10000;';
        
        const content = document.createElement('div');
        content.style.cssText = 'background: white; padding: 20px; border-radius: 8px; max-width: 600px; max-height: 80vh; overflow-y: auto;';
        
        const title = document.createElement('h2');
        title.textContent = `Edit Row - GCID: ${gcid}`;
        
        const form = document.createElement('div');
        form.innerHTML = formHtml;
        
        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = 'margin-top: 20px; display: flex; gap: 10px; justify-content: flex-end;';
        
        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Save Changes';
        saveBtn.style.cssText = 'background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;';
        saveBtn.onclick = () => saveRowChanges(rowIdx, modal);
        
        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Cancel';
        closeBtn.style.cssText = 'background-color: #f44336; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;';
        closeBtn.onclick = () => modal.remove();
        
        buttonContainer.appendChild(saveBtn);
        buttonContainer.appendChild(closeBtn);
        
        content.appendChild(title);
        content.appendChild(form);
        content.appendChild(buttonContainer);
        modal.appendChild(content);
        document.body.appendChild(modal);
    }
    
    function saveRowChanges(rowIdx, modal) {
        const rowData = window.rowDataMap[rowIdx];
        const updates = {};

        // Collect edited fields (skip internal source markers)
        for (const key of Object.keys(rowData)) {
            if (key === '_source_file' || key === '_source_index') continue;
            const inputElement = document.getElementById(`edit_${key}`);
            if (inputElement) {
                updates[key] = inputElement.value;
            }
        }

        // Include source file and source index so server can update the correct CSV
        const source_file = rowData['_source_file'];
        const source_index = rowData['_source_index'];

        fetch('/api/update_row', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({rowIdx: rowIdx, updates: updates, source_file: source_file, source_index: source_index})
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Row saved successfully!');
                modal.remove();
                location.reload();
            } else {
                alert('Error saving row: ' + data.error);
            }
        })
        .catch(error => {
            alert('Error: ' + error);
        });
    }
    </script>
    '''
    
    # Save map
    golf_map.save(output_file)

    # Inject the row data and JavaScript into the HTML file
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Store row data as JavaScript object (from unique_locations)
    row_data_script = f'<script>window.rowDataMap = {json.dumps(row_data_map, default=str)};</script>'

    # Inject scripts before closing body tag
    html_content = html_content.replace('</body>', f'{row_data_script}\n{edit_js}\n</body>')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Map saved to: {output_file}")

    return golf_map


def main(csv_paths, output_file: str = "golf_courses_map.html", serve=True, port=8000):
    """
    Main function to generate golf course map.
    
    Args:
        csv_path: Path to the CSV file containing golf course data
        output_file: Name of the output HTML file
        serve: Whether to start a local web server (default: True)
        port: Port number for the web server (default: 8000)
    """
    global csv_path_global
    # Allow csv_paths to be a single string or a list
    if isinstance(csv_paths, (list, tuple)):
        input_paths = csv_paths
    else:
        input_paths = [csv_paths]

    csv_path_global = [str(p) for p in input_paths]

    # Load data
    print("Loading golf course data from:")
    for p in input_paths:
        print(f" - {p}")

    # Combine multiple CSVs
    df = load_multiple_csvs(input_paths)
    
    # Get unique locations
    print("\nExtracting unique locations...")
    unique_locations = get_unique_locations(df)
    
    # Create map
    print("\nGenerating map...")

    # Save the HTML one directory above the `images/` folder (i.e., repo root next to `images/`)
    repo_root = Path(__file__).parent.resolve()
    map_file_path = repo_root / output_file
    create_canada_map(df, unique_locations, None, str(map_file_path))

    if serve:
        # Serve from repository root so the HTML can reference the `images/` folder as a sibling
        serve_dir = repo_root
        os.chdir(serve_dir)

        # Start local server
        server_address = ('localhost', port)
        httpd = HTTPServer(server_address, CustomRequestHandler)

        # Run server in a background thread
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        map_url = f"http://localhost:{port}/{output_file}"
        print(f"\n✓ Local server started on {map_url}")
        print("The map will now open in your browser.")
        print("Edit features are now enabled! Close the server when done.")

        # Open browser
        webbrowser.open(map_url)

        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            httpd.shutdown()
    else:
        print("\nDone!")


if __name__ == "__main__":
    # Default path to the CSV file
    csv_file_canada = Path(__file__).parent / "data" / "canada" / "Fully_Matched_Golf_Courses.csv"
    csv_file_usa = Path(__file__).parent / "data" / "usa" / "combined.csv"
    files = (str(csv_file_canada))
    
    main(files)
