import numpy as np
import pandas as pd

# Original suppliers table
suppliers = pd.DataFrame(
    {
        "supplier_id": range(1, 6),
        "supplier_name": ["Avis", "Hertz", "Enterprise", "Sixt", "Budget"],
    }
)

# City supplier patterns
city_supplier_patterns = {
    "New York": {
        "Enterprise": [1, 2, 3, 4],
        "Hertz": [1, 2, 3, 4],
        "Avis": [1, 2, 3, 4],
        "Budget": [1, 2, 3],
        "Sixt": [1],
    },
    "Los Angeles": {
        "Enterprise": [5, 6, 7, 8],
        "Hertz": [5, 6, 7, 8],
        "Avis": [5, 6, 7, 8],
        "Budget": [5, 6, 7],
        "Sixt": [5, 6],
    },
    "Chicago": {
        "Enterprise": [9, 10, 11, 12],
        "Hertz": [9, 10, 11, 12],
        "Avis": [9, 10, 11],
        "Budget": [9, 10, 11],
        "Sixt": [9],
    },
    "Atlanta": {
        "Enterprise": [13, 14, 15, 16],
        "Hertz": [13, 14, 15, 16],
        "Avis": [13, 14, 15],
        "Budget": [13, 14],
        "Sixt": [13],
    },
    "Houston": {
        "Enterprise": [17, 18, 19, 20],
        "Hertz": [17, 18, 19],
        "Avis": [17, 18, 19],
        "Budget": [17, 18, 19],
        "Sixt": [17],
    },
    "Miami": {
        "Enterprise": [21, 22, 23, 24],
        "Hertz": [21, 22, 23, 24],
        "Avis": [21, 22, 23, 24],
        "Budget": [21, 22, 23],
        "Sixt": [21],
    },
}


# Method 1: Using pandas operations without explicit for loops in DataFrame creation
def method1_pandas_explode():
    # Create a list of records more efficiently
    records = []
    supplier_id_map = dict(zip(suppliers["supplier_name"], suppliers["supplier_id"]))

    for city, supplier_data in city_supplier_patterns.items():
        for supplier_name, location_ids in supplier_data.items():
            supplier_id = supplier_id_map[supplier_name]
            city_records = [
                {
                    "supplier_id": supplier_id,
                    "location_id": loc_id,
                    "supplier_name": supplier_name,
                    "city": city,
                }
                for loc_id in location_ids
            ]
            records.extend(city_records)

    return pd.DataFrame(records)


# Method 2: Using list comprehension (most Pythonic)
def method2_list_comprehension():
    supplier_id_map = dict(zip(suppliers["supplier_name"], suppliers["supplier_id"]))

    data = [
        (supplier_id_map[supplier_name], location_id, supplier_name, city)
        for city, supplier_data in city_supplier_patterns.items()
        for supplier_name, location_ids in supplier_data.items()
        for location_id in location_ids
    ]

    return pd.DataFrame(
        data, columns=["supplier_id", "location_id", "supplier_name", "city"]
    )


# Method 3: Using pandas melt/stack operations
def method3_pandas_reshape():
    # First, create a MultiIndex DataFrame from the patterns
    data_for_df = []
    for city, suppliers_dict in city_supplier_patterns.items():
        for supplier_name, locations in suppliers_dict.items():
            data_for_df.append(
                {"city": city, "supplier_name": supplier_name, "locations": locations}
            )

    df_temp = pd.DataFrame(data_for_df)

    # Explode the locations list to get individual rows
    df_exploded = df_temp.explode("locations")
    df_exploded.rename(columns={"locations": "location_id"}, inplace=True)

    # Merge with suppliers to get supplier_id
    df_result = df_exploded.merge(suppliers, on="supplier_name")

    # Reorder columns
    return df_result[["supplier_id", "location_id", "supplier_name", "city"]]


# Method 4: Using numpy for even more efficiency
def method4_numpy_approach():
    supplier_id_map = dict(zip(suppliers["supplier_name"], suppliers["supplier_id"]))

    # Pre-calculate total number of rows
    total_rows = sum(
        len(locations)
        for city_data in city_supplier_patterns.values()
        for locations in city_data.values()
    )

    # Pre-allocate arrays
    supplier_ids = np.zeros(total_rows, dtype=int)
    location_ids = np.zeros(total_rows, dtype=int)
    supplier_names = []
    cities = []

    idx = 0
    for city, supplier_data in city_supplier_patterns.items():
        for supplier_name, locations in supplier_data.items():
            n_locations = len(locations)
            supplier_ids[idx : idx + n_locations] = supplier_id_map[supplier_name]
            location_ids[idx : idx + n_locations] = locations
            supplier_names.extend([supplier_name] * n_locations)
            cities.extend([city] * n_locations)
            idx += n_locations

    return pd.DataFrame(
        {
            "supplier_id": supplier_ids,
            "location_id": location_ids,
            "supplier_name": supplier_names,
            "city": cities,
        }
    )


# Method 5: Using numpy with list comprehensions (no explicit for loops)
def method5_numpy_listcomp():
    supplier_id_map = dict(zip(suppliers["supplier_name"], suppliers["supplier_id"]))

    # Create lists of numpy arrays for each supplier-location group
    supplier_id_arrays = [
        np.full(len(locations), supplier_id_map[supplier_name])
        for city, supplier_data in city_supplier_patterns.items()
        for supplier_name, locations in supplier_data.items()
    ]

    location_id_arrays = [
        np.array(locations)
        for city, supplier_data in city_supplier_patterns.items()
        for supplier_name, locations in supplier_data.items()
    ]

    # Concatenate all arrays
    supplier_ids = np.concatenate(supplier_id_arrays)
    location_ids = np.concatenate(location_id_arrays)

    # Create flattened lists for string columns
    supplier_names = [
        supplier_name
        for city, supplier_data in city_supplier_patterns.items()
        for supplier_name, locations in supplier_data.items()
        for _ in locations
    ]

    cities = [
        city
        for city, supplier_data in city_supplier_patterns.items()
        for supplier_name, locations in supplier_data.items()
        for _ in locations
    ]

    return pd.DataFrame(
        {
            "supplier_id": supplier_ids,
            "location_id": location_ids,
            "supplier_name": supplier_names,
            "city": cities,
        }
    )


# Test all methods
if __name__ == "__main__":
    print("Testing different vectorization methods:\n")

    # Method 1
    df1 = method1_pandas_explode()
    print("Method 1 - Pandas with efficient records:")
    print(f"Shape: {df1.shape}")
    print(df1.head())

    # Method 2
    df2 = method2_list_comprehension()
    print("\n\nMethod 2 - List comprehension:")
    print(f"Shape: {df2.shape}")
    print(df2.head())

    # Method 3
    df3 = method3_pandas_reshape()
    print("\n\nMethod 3 - Pandas reshape operations:")
    print(f"Shape: {df3.shape}")
    print(df3.head())

    # Method 4
    df4 = method4_numpy_approach()
    print("\n\nMethod 4 - NumPy approach:")
    print(f"Shape: {df4.shape}")
    print(df4.head())

    # Method 5
    df5 = method5_numpy_listcomp()
    print("\n\nMethod 5 - NumPy with list comprehensions:")
    print(f"Shape: {df5.shape}")
    print(df5.head())

    # Verify all methods produce the same result
    print("\n\nVerification:")
    print(f"Method 1 == Method 2: {df1.equals(df2)}")
    print(
        f"Method 2 == Method 3: {df2.sort_values(['supplier_id', 'location_id']).reset_index(drop=True).equals(df3.sort_values(['supplier_id', 'location_id']).reset_index(drop=True))}"
    )
    print(f"Method 2 == Method 4: {df2.equals(df4)}")
    print(f"Method 2 == Method 5: {df2.equals(df5)}")
    print(f"Method 4 == Method 5: {df4.equals(df5)}")

    # Performance comparison
    import timeit

    print("\n\nPerformance comparison (1000 iterations):")
    for method_name, method_func in [
        ("Method 1 - Pandas records", method1_pandas_explode),
        ("Method 2 - List comprehension", method2_list_comprehension),
        ("Method 3 - Pandas reshape", method3_pandas_reshape),
        ("Method 4 - NumPy", method4_numpy_approach),
        ("Method 5 - NumPy + list comp", method5_numpy_listcomp),
    ]:
        time_taken = timeit.timeit(method_func, number=1000)
        print(f"{method_name}: {time_taken:.4f} seconds")
