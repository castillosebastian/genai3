import json

def build_query_filter_0(json_object):
    try:
        # Initialize a list to hold filter strings for each company
        filters = []

        # Get the list of companies or an empty list if None
        companies = json_object.get('company_name', []) if json_object.get('company_name') is not None else []

        # Iterate over each company
        for company in companies:
            # Initialize components of the filter for each company
            filter_components = []

            # Add company filter
            filter_components.append(f"Description eq '{company}'")

            # Check and append country if it exists
            if json_object.get('country') and json_object['country'] is not None:
                country = json_object['country'][0]  # Assuming you want to use the first item
                filter_components.append(f"Country eq '{country}'")

            # Check and append dates if they exist
            if json_object.get('dates') and json_object['dates'] is not None:
                # Assuming you want to use the year from the first date
                year = json_object['dates'][0][:4]  # Extracts the year part from the date
                filter_components.append(f"AdditionalMetadata eq {year}")

            # Join all components with 'and' for the current company
            filter_query = " and ".join(filter_components)

            # Add the filter query for the current company to the list
            filters.append(filter_query)

        return filters

    except Exception as e:
        # Handle any exception that occurs during processing
        return f"Error building query filter: {e}"

def build_query_filter(json_object):
    try:
        # Initialize a list to hold filter strings
        filters = []

        # Get the list of companies or use a list with None if company_name is None
        companies = json_object.get('company_name', [None]) if json_object.get('company_name') is not None else [None]

        # Check and prepare country and dates outside the loop
        country = json_object['country'][0] if json_object.get('country') else None
        year = json_object['dates'][0][:4] if json_object.get('dates') else None

        # Loop through each company
        for company in companies:
            # Initialize components of the filter for this company
            filter_components = []

            # Add company filter if company exists
            if company:
                filter_components.append(f"Description eq '{company}'")

            # Add country filter if country exists
            if country:
                filter_components.append(f"Country eq '{country}'")

            # Add date filter if dates exist
            if year:
                filter_components.append(f"AdditionalMetadata eq {year}")

            # Check if any filter component was added for this company/criteria
            if filter_components:
                # Join all components with 'and'
                filter_query = " and ".join(filter_components)
                filters.append(filter_query)

        # Check if any filter was created
        if not filters:
            return False

        return filters

    except Exception as e:
        # Handle any exception that occurs during processing
        return [f"Error building query filter: {e}"]


# Example usage  
json_data = {'company_name': ['BestBuy', 'AnotherCompany'], 'country': ['US'], 'dates': ['2019-01-01', '2019-12-31']}
filter_queries = build_query_filter(json_data)
print(len(filter_queries))
for query in filter_queries:
    print(query)

test_json_data = [
    {'company_name': ['BestBuy'], 'country': ['US'], 'dates': ['2019-01-01', '2019-12-31']},
    {'company_name': ['Apple'], 'country': ['US'], 'dates': None},
    {'company_name': None, 'country': ['UK'], 'dates': ['2020-01-01', '2020-12-31']},
    {'company_name': ['Google', 'Amazon', 'Microsoft'], 'country': ['US', 'IE'], 'dates': ['2021-01-01', '2021-12-31']},
    {'company_name': ['Tesla'], 'country': None, 'dates': ['2022-01-01', '2022-12-31']},
    {'company_name': ['IBM'], 'country': ['CA'], 'dates': None},
    {'company_name': None, 'country': None, 'dates': None},
    {'company_name': ['Samsung'], 'country': ['KR'], 'dates': ['2018-01-01', '2018-12-31']},
    {'company_name': ['Sony', 'LG'], 'country': ['JP', 'KR'], 'dates': None},
    {'company_name': ['Dell'], 'country': ['US'], 'dates': ['2019-06-01', '2019-11-30']},
    {'company_name': ['HP', 'Lenovo'], 'country': None, 'dates': ['2020-02-01', '2020-10-31']},
    {'company_name': None, 'country': ['DE'], 'dates': ['2021-03-01', '2021-09-30']},
    {'company_name': ['Xiaomi'], 'country': ['CN'], 'dates': None},
    {'company_name': ['Asus', 'Acer'], 'country': ['TW'], 'dates': ['2017-01-01', '2017-12-31']},
    {'company_name': ['Huawei'], 'country': ['CN', 'US'], 'dates': ['2016-04-01', '2016-12-31']},
    {'company_name': ['Facebook'], 'country': None, 'dates': None},
    {'company_name': ['Intel', 'AMD'], 'country': ['US'], 'dates': ['2022-05-01', '2022-12-31']},
    {'company_name': ['Nvidia'], 'country': ['US', 'CN'], 'dates': None},
    {'company_name': None, 'country': ['FR'], 'dates': ['2018-07-01', '2018-12-31']},
    {'company_name': ['Adobe'], 'country': ['US'], 'dates': ['2019-01-01', '2019-12-31']},
    {'company_name': ['Spotify'], 'country': ['SE'], 'dates': None},
    {'company_name': ['Twitter'], 'country': None, 'dates': ['2020-06-01', '2020-12-31']},
    {'company_name': ['Oracle', 'Salesforce'], 'country': ['US', 'IN'], 'dates': ['2021-07-01', '2021-12-31']},
    {'company_name': ['PayPal'], 'country': ['US'], 'dates': None},
    {'company_name': None, 'country': ['IT'], 'dates': ['2019-02-01', '2019-08-31']},
    {'company_name': ['Zoom'], 'country': ['US'], 'dates': ['2022-03-01', '2022-09-30']},
    {'company_name': ['Uber'], 'country': ['US', 'GB'], 'dates': None},
    {'company_name': None, 'country': ['MX'], 'dates': ['2020-04-01', '2020-10-31']},
    {'company_name': ['Lyft'], 'country': None, 'dates': ['2021-01-01', '2021-06-30']},
    {'company_name': ['Airbnb'], 'country': ['US'], 'dates': None},
]


# Function to run test cases
def test_function_execution(test_json_data):
    success_count = 0
    failure_count = 0

    for i, json_input in enumerate(test_json_data):
        try:
            result = build_query_filter(json_input)
            if result:
                print(f"Test {i+1}: Success - Result: {result}")
                success_count += 1
            else:
                print(result)
                failure_count += 1
        except Exception as e:
            print(f"Test {i+1}: Failed with error - {e}")
            failure_count += 1

    print("\n")
    print(f"Total tests: {len(test_json_data)}")
    print(f"Success: {success_count}")
    print(f"Failures/Empty Outputs: {failure_count}")

# Run the test cases
    
if __name__ == "__main__":   
    test_function_execution(test_json_data)

