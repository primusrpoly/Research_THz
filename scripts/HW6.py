import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def scrape_website(term_url, base_url):
    # Initialize a new Excel writer object
    writer = pd.ExcelWriter('terms.xlsx', engine='xlsxwriter')
    
    # Initialize the unique ID counter
    unique_id = 1

    # Send a GET request to the term_url
    response = requests.get(term_url)
    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'lxml')

    # Find the dropdown menu by its name
    select = soup.find('select', {'name': 'term_in'})

    # Get all options in the dropdown menu
    term_options = select.find_all('option')
    term_form = soup.find('form')
    term_action = term_form['action']
    absolute_term_url = base_url + term_action

    # For each option in the term dropdown menu
    for term_option in term_options:
        option_value = term_option['value']
        
        term_data = {
            'term_in': option_value,
        }

        response = requests.post(absolute_term_url, data=term_data)

        # Get the term
        term = term_option.text
        # Get the numeric term
        numeric_term = term_option['value']

        # Print the term and numeric term
        print(f'Term: {term}, Numeric Term: {numeric_term}')

        # Parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'lxml')

        # Find the dropdown menu by its name
        dropdown = soup.find('select', {'name': 'disc_in'})

        # Get all options in the dropdown menu
        discipline_options = dropdown.find_all('option')
        disc_form = soup.find('form')
        disc_action = disc_form['action']
        abs_disc_url = base_url + disc_action

        # Initialize a DataFrame to collect all disciplines for the current term
        term_df = pd.DataFrame()

        # For each option in the discipline dropdown menu
        for discipline_option in discipline_options:
            disc_option_value = discipline_option['value']
            #print(disc_option_value)
            disc_data = {
                'term_in': option_value,
                'disc_in': disc_option_value,
            }

            response = requests.post(abs_disc_url, data=disc_data)

            # Get the discipline
            discipline = discipline_option.text

            # Print the discipline
            print(f'Scraping {term} {discipline}...')

            # Parse the HTML content of the page with BeautifulSoup
            soup = BeautifulSoup(response.text, 'lxml')

            # Find the table with the data
            tables = soup.find_all('table')
            if len(tables) > 2:
                table = tables[2]
            else:
                print(f"No valid table found for discipline: {discipline}")
                continue

            # Parse the table with pandas
            data = pd.read_html(StringIO(str(table)), header=0)[0]

            # Add columns for the discipline, term, and unique ID
            data['Discipline'] = discipline
            data['Term'] = numeric_term
            data['ID'] = range(unique_id, unique_id + len(data))
            unique_id += len(data)

            # Append the data to the term DataFrame
            term_df = pd.concat([term_df, data], ignore_index=True)

        # Write the term DataFrame to the Excel file
        sheet_name = numeric_term[:31]
        invalid_chars = '[]:*?/\\'
        for char in invalid_chars:
            sheet_name = sheet_name.replace(char, '')
        term_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel file
    writer.close()

# The URLs of the website to scrape
term_url = "https://banner.sunypoly.edu/pls/prod/swssschd.P_SelDefSchTerm?proc_name=swssschd.P_SelDisc"
base_url = "https://banner.sunypoly.edu"

scrape_website(term_url, base_url)
