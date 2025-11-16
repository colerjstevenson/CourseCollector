from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

class PostalCodeLookup:
    def __init__(self, user_agent="postal_code_lookup"):
        self.geolocator = Nominatim(user_agent=user_agent)

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


if __name__ == "__main__":
    # Example usage
    lookup = PostalCodeLookup()
    lat = 40.7128
    lon = -74.0060

    print("Postal code:", lookup.get_postal_code(lat, lon))
