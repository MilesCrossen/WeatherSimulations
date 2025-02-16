import folium

#station data format -> name, type, capacity, latitude, longitude
stations = [
    ("Athenry", "Solar", 71.16, 53.283333, -8.75),
    ("Ballyhaise", "Solar", 1199.47, 54.05, -7.266667),
    ("Carlow Oakpark", "Wind", 1723.27, 52.85, -6.933333),
    ("Carlow Oakpark", "Solar", 404.35, 52.85, -6.933333),
    ("Claremorris", "Solar", 163.81, 53.716667, -8.983333),
    ("Fermoy Moorepark", "Solar", 1108.69, 52.133333, -8.283333),
    ("Finner", "Solar", 581.01, 54.483333, -8.233333),
    ("Gurteen", "Solar", 946.15, 53.016667, -8.1),
    ("Johnstown Castle", "Wind", 94.12, 52.3, -6.5),
    ("Johnstown Castle", "Solar", 140.79, 52.3, -6.5),
    ("Mace Head", "Wind", 5139.33, 53.333333, -9.9),
    ("Mace Head", "Solar", 2.28, 53.333333, -9.9),
    ("Mount Dillon", "Wind", 202.38, 53.766667, -8.116667),
    ("Mount Dillon", "Solar", 324.70, 53.766667, -8.116667),
    ("Mullingar", "Solar", 598.81, 53.533333, -7.333333),
    ("Roches Point", "Solar", 325.00, 51.783333, -8.266667),
    ("Sherkin Island", "Wind", 525.94, 51.466667, -9.416667),
    ("Sherkin Island", "Solar", 852.53, 51.466667, -9.416667),
]

#create map centred around Ireland
m = folium.Map(location=[53.5, -8.0], zoom_start=7)

#add stations to map
for station, energy_type, capacity, lat, lon in stations:
    color = "blue" if energy_type == "Wind" else "orange"

    folium.Marker(
        location=[lat, lon],
        popup=f"<b>{station}</b><br>Type: {energy_type}<br>Capacity: {capacity} MW",
        icon=folium.Icon(color=color),
    ).add_to(m)

#save map to html file
m.save("renewable_energy_map.html")

print("Interactive map saved as 'renewable_energy_map.html'.")