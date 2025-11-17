from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

import csv
from collections import defaultdict

class PostalCodeLookup:
    def __init__(self, user_agent="postal_code_lookup"):
        self.geolocator = Nominatim(user_agent=user_agent)
        
    
        
    def greedy_match_by_postal(coords_csv, info_csv, output_csv):
        # Load coordinates table (A)
        coords_by_postal = defaultdict(list)
        print("Loading coordinates...")
        with open(coords_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                postal = row["postal_code"].replace(" ", "").upper()
                if postal:
                    coords_by_postal[postal].append(row)
                else:
                    print(f"  Warning: could not find postal code for {row['gcid']}. Skipping.")

        # Load info table (B)
        info_by_postal = defaultdict(list)
        print("Loading golf link...")
        with open(info_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                postal = row["Zip"].replace(" ", "").upper()
                if postal in coords_by_postal:
                    info_by_postal[postal].append(row)

        # Prepare output
        fieldnames = [
                    "gcid",
                    "latitude",
                    "longitude",
                    "area_m2"
                    "AccessType",
                    "Address",
                    "City",
                    "postal_code",
                    "CourseName",
                    "NumHoles",
                    "Par",
                    "Phone",
                    "State",
                    "Yardage",
                    "Zip",
                    "established",
                    "url",
                    "website"
                    ]
        print("Writing output...")
        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for postal in set(coords_by_postal):
                print(f"Processing postal code: {postal}")
                coords_list = coords_by_postal.get(postal, [])
                info_list = info_by_postal.get(postal, [])

                

                lenA = len(coords_list)
                lenB = len(info_list)
                
                if lenA == 0:
                    print(f"  No coordinates found for postal code {postal}, skipping.")
                    continue
                
                if lenB == 0:
                    print(f"  No info found for postal code {postal}, adding NOMATCH entries.")
                    info_list.append({
                        "CourseName": "NOMATCH",
                        "NumHoles": "NOMATCH",
                        "Par": "NOMATCH",
                        "Phone": "NOMATCH",
                        "Address": "NOMATCH",
                        "City": "NOMATCH",
                        "State": "NOMATCH",
                        "Yardage": "NOMATCH",
                        "established": "NOMATCH",
                        "url": "NOMATCH",
                        "website": "NOMATCH"
                    })
                    lenB = 1
                    match_type = "no_info_match"

                n = max(lenA, lenB)
                for i in range(n):
                    a = coords_list[i%(lenA)]  # wrap around if needed
                    b = info_list[i%(lenB)]  # wrap around if needed
                    
                    if not match_type:
                        if lenA == 1 and lenB == 1:
                            match_type = "unique"
                        elif lenA > lenB:
                            match_type = "multiple_coords_more"
                        elif lenB > lenA:
                            match_type = "multiple_info_more"
                        else:
                            match_type = "multiple"
                    
                    writer.writerow({
                        "postal_code": postal,
                        "gcid": a["gcid"],
                        "latitude": a["lat"],
                        "longitude": a["lon"],
                        "area_m2": a["area_m2"],
                        "courseName": b["CourseName"],
                        "NumHoles": b["NumHoles"],
                        "Par": b["Par"],
                        "Phone": b["Phone"],
                        "Address": b["Address"],
                        "City": b["City"],
                        "Region": b["State"],
                        "Yardage": b["Yardage"],
                        "established": b["established"],
                        "url": b["url"],
                        "website": b["website"],
                        "match_type": match_type
                    })

               

    def get_postal_code(self, latitude, longitude):
        """
        Returns the postal code (ZIP code in the US) for a given lat/lon pair.

        Returns:
            postal_code (str) or None if not found.
        """
        try:
            location = self.geolocator.reverse((latitude, longitude), addressdetails=True)
        except (GeocoderTimedOut, GeocoderUnavailable):
            return None

        if not location or "address" not in location.raw:
            return None

        address = location.raw["address"]

        # Postal code fields differ slightly by country but "postcode" is universal
        postal_code = address.get("postcode")
        return postal_code
    
    def add_postal_codes(self, input_csv):
        # Load input CSV and add postal codes
        output_rows = []
        print("Adding postal codes...")
        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            number_of_rows = sum(1 for _ in open(input_csv)) - 1  # subtract header
            for i, row in enumerate(reader):
                if "postal_code" in row and row["postal_code"].strip():
                    print(f"Skipping GCID: {row['gcid']} (already has postal code)")
                    output_rows.append(row)
                    continue
                print(f"[{i}/{number_of_rows}] Getting Postal Code: {row['gcid']}: {row['name']}")
                lat = float(row["lat"])
                lon = float(row["lon"])
                postal_code = self.get_postal_code(lat, lon)
                row["postal_code"] = postal_code if postal_code else "NOMATCH"
                output_rows.append(row)
                print(f"  Found postal code: {row['postal_code']}")

        # Write output CSV
        output_csv = input_csv
        fieldnames = list(output_rows[0].keys())
        print(f"Writing output to {output_csv}...")
        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)
        print("postal codes added.")


if __name__ == "__main__":
    COUNTRY = "canada"
    # Example usage
    lookup = PostalCodeLookup()
    lookup.add_postal_codes(f"data/{COUNTRY}/combined.csv")
    lookup.greedy_match_by_postal(f"data/{COUNTRY}/combined.csv", "data/golfLinkData.csv", f"data/{COUNTRY}/Fully_Matched_Golf_Courses.csv")
