import uiautomation as auto
import subprocess
import time

def extract_ui_elements_to_xml(element, depth=0):
    """ Recursively extract the UI elements in XML format """
    xml_output = f"{'  ' * depth}<Element Type='{element.ControlTypeName}' Name='{element.Name}'>\n"
    
    # Iterate through all child elements recursively
    for child in element.GetChildren():
        xml_output += extract_ui_elements_to_xml(child, depth + 1)
    
    xml_output += f"{'  ' * depth}</Element>\n"
    return xml_output

def find_and_click_by_name(element, target_name):
    """ Recursively search for an element by name and click it """
    if element.Name == target_name:
        print(f"Clicking on element: {element.Name}")
        element.Click()
        return True

    # Search through child elements
    for child in element.GetChildren():
        if find_and_click_by_name(child, target_name):
            return True

    return False

# Step 1: Open Notepad
subprocess.Popen('notepad.exe')

# Give Notepad some time to open
time.sleep(2)

# Step 2: Get the Notepad window and extract its elements
notepad_window = auto.WindowControl(searchDepth=1, Name="Untitled - Notepad")
if not notepad_window.Exists():
    print("Notepad window not found")
    exit()

# Extract UI elements in XML format
xml_output = extract_ui_elements_to_xml(notepad_window)
print(xml_output)  # Output XML structure to understand it

# Step 3: Find and click the "File" menu
if not find_and_click_by_name(notepad_window, "File"):
    print("File menu not found")
    exit()

# Give the file menu time to open
time.sleep(1)

# Step 4: Find and click the "Print" option from the File menu
if not find_and_click_by_name(notepad_window, "Print..."):
    print("Print option not found")
else:
    print("Successfully clicked 'Print'")

