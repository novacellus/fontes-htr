#!/usr/bin/env python3
"""
Enhanced script to process work directories containing PAGE XML files and images.
For each work directory, this script:
1. Creates a corresponding directory in the target location
2. Copies all JPG/TIFF image files
3. Removes TranskribusMetadata and logs it
4. Converts PageXML to latest format using JPageConverter1.5
5. (Optional) Converts updated PAGE XML files to ALTO XML using ocrd-page-to-alto
"""

import os
import sys
import shutil
import subprocess
import argparse
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def setup_logging(log_file):
    """Set up logging to both console and file."""
    logger = logging.getLogger('page_processor')
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_work_directories(src_dir):
    """List all directories within the source directory."""
    return [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]


def collect_files(work_dir):
    """Collect image and PAGE XML files from a work directory."""
    image_files = []
    xml_files = []

    # Files to exclude - these patterns won't be processed
    exclude_patterns = [
        'doc.xml',        # Documentation XML
        'temp',           # Any file containing 'temp'
        'metadata.xml',   # Metadata files
        'config.xml',     # Configuration files
    ]

    # Walk through the directory
    for root, _, files in os.walk(work_dir):
        for file in files:
            # Skip files matching exclude patterns
            if any(pattern in file.lower() for pattern in exclude_patterns):
                continue

            file_path = os.path.join(root, file)

            # Check if the file is an image (JPG or TIFF)
            if file.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff')):
                image_files.append(file_path)

            # Check if the file is a PAGE XML file
            elif file.lower().endswith('.xml'):
                # Here we could add more checks to ensure it's a PageXML file
                # For example, we could check for the presence of PcGts element
                # This would require opening and parsing each XML file
                xml_files.append(file_path)

    return image_files, xml_files


def remove_transkribus_metadata(input_xml, output_xml, logger):
    """
    Remove TranskribusMetadata element from PAGE XML and log it.

    Returns True if successful, False otherwise.
    """
    try:
        # Register namespace
        ET.register_namespace('', "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")

        # Parse the XML
        tree = ET.parse(input_xml)
        root = tree.getroot()

        # Find the Metadata element
        metadata_elem = None
        for elem in root.iter():
            if elem.tag.endswith('Metadata'):
                metadata_elem = elem
                break

        if metadata_elem is None:
            logger.warning(f"No Metadata element found in {input_xml}")
            # Just copy the file
            shutil.copy2(input_xml, output_xml)
            return True

        # Find TranskribusMetadata within Metadata
        transkribus_elem = None
        for elem in metadata_elem:
            if elem.tag.endswith('TranskribusMetadata'):
                transkribus_elem = elem
                break

        if transkribus_elem is not None:
            # Log the TranskribusMetadata information
            logger.info(f"Removed TranskribusMetadata from {input_xml}: {ET.tostring(transkribus_elem, encoding='unicode')}")

            # Remove the element
            metadata_elem.remove(transkribus_elem)

            # Write the modified XML
            tree.write(output_xml, encoding='UTF-8', xml_declaration=True)
            return True
        else:
            logger.info(f"No TranskribusMetadata found in {input_xml}")
            # Just copy the file
            shutil.copy2(input_xml, output_xml)
            return True

    except Exception as e:
        logger.error(f"Error processing {input_xml}: {e}")
        return False


def convert_to_latest_page_format(input_xml, output_xml, logger):
    """
    Convert PAGE XML to latest format using JPageConverter1.5.

    Returns True if successful, False otherwise.
    """
    # Build the command
    converter_jar = "./JPageConverter1.5/PageConverter.jar"
    cmd = [
        "java",
        "-jar",
        converter_jar,
        "-source-xml",
        input_xml,
        "-target-xml",
        output_xml,
        "-convert-to",
        "LATEST"
    ]

    # Execute the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Converted {input_xml} to latest PAGE format at {output_xml}")

        # Verify the file was actually created
        if os.path.exists(output_xml) and os.path.getsize(output_xml) > 0:
            return True
        else:
            logger.error(f"Conversion seemed successful but output file {output_xml} does not exist or is empty")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_xml} to latest PAGE format: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during PAGE conversion: {e}")
        return False


def convert_page_to_alto(page_xml_path, target_dir, logger):
    """Convert a PAGE XML file to ALTO format using ocrd-page-to-alto."""
    # Determine the output filename (same basename but in the target directory)
    filename = os.path.basename(page_xml_path)
    base_name = os.path.splitext(filename)[0]
    alto_filename = f"{base_name}.xml"
    output_path = os.path.join(target_dir, alto_filename)

    # Build the command
    cmd = [
        "page-to-alto",
        "--no-check-border",
        "--no-check-words",
        "-O", output_path,
        page_xml_path
    ]

    # Execute the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Converted {page_xml_path} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {page_xml_path} to ALTO: {e}")
        logger.error(f"Command output: {e.stdout if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"Command error: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during ALTO conversion: {e}")
        return False


def find_matching_image(xml_file, image_files):
    """
    Check if there's a matching image file for the given XML file.
    Returns the matching image file path if found, None otherwise.
    """
    # Extract base name without extension
    xml_basename = os.path.splitext(os.path.basename(xml_file))[0]

    # Try to find matching image with same base name but different extension
    for img_file in image_files:
        img_basename = os.path.splitext(os.path.basename(img_file))[0]
        if xml_basename == img_basename:
            return img_file

    # Try to find matching image with base name contained in the XML file name
    # This handles cases where XML might be named like "page_0001_alto.xml"
    # while image is "page_0001.jpg"
    for img_file in image_files:
        img_basename = os.path.splitext(os.path.basename(img_file))[0]
        if img_basename in xml_basename:
            return img_file

    # No matching image found
    return None

def is_page_xml(xml_file, logger):
    """
    Check if an XML file is actually a PAGE XML file by looking for specific elements.
    Returns True if it appears to be a PAGE XML, False otherwise.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Check if this is a PcGts element or has a namespace that suggests PAGE format
        root_tag = root.tag.lower()
        is_page = ('pcgts' in root_tag) or ('pagecontent' in root_tag)

        # Check for typical PAGE elements
        has_page_elements = False
        for elem in root.iter():
            if any(key in elem.tag.lower() for key in ['page', 'textregion', 'textline', 'coords']):
                has_page_elements = True
                break

        return is_page or has_page_elements
    except Exception as e:
        logger.warning(f"Error checking if {xml_file} is a PAGE XML: {e}")
        return False
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Check if this is a PcGts element or has a namespace that suggests PAGE format
        root_tag = root.tag.lower()
        is_page = ('pcgts' in root_tag) or ('pagecontent' in root_tag)

        # Check for typical PAGE elements
        has_page_elements = False
        for elem in root.iter():
            if any(key in elem.tag.lower() for key in ['page', 'textregion', 'textline', 'coords']):
                has_page_elements = True
                break

        return is_page or has_page_elements
    except Exception as e:
        logger.warning(f"Error checking if {xml_file} is a PAGE XML: {e}")
        return False

def process_work_directory(src_dir, work_name, target_base_dir, logger, convert_to_alto=False):
    """Process a single work directory."""
    work_dir = os.path.join(src_dir, work_name)
    target_dir = os.path.join(target_base_dir, work_name)

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Processing work directory: {work_name}")

    # Collect image and XML files
    image_files, xml_files = collect_files(work_dir)

    logger.info(f"Found {len(image_files)} image files and {len(xml_files)} XML files to process")

    # Copy image files
    for img_file in image_files:
        try:
            target_file = os.path.join(target_dir, os.path.basename(img_file))
            shutil.copy2(img_file, target_file)
            logger.info(f"Copied {img_file} to {target_file}")
        except Exception as e:
            logger.error(f"Failed to copy image file {img_file}: {e}")
            print(f"ERROR: Failed to copy image file {img_file}: {e}")

    # Process XML files
    processed_count = 0
    skipped_count = 0
    no_image_count = 0
    error_count = 0

    # Create a temp directory for this work directory's processing
    with tempfile.TemporaryDirectory(prefix="page_processor_") as work_temp_dir:
        logger.info(f"Created temporary directory for processing: {work_temp_dir}")

        for xml_file in xml_files:
            try:
                # Check if there's a matching image file
                matching_image = find_matching_image(xml_file, image_files)
                if not matching_image:
                    msg = f"WARNING: No matching image found for XML file: {xml_file}"
                    logger.warning(msg)
                    print(msg)  # Echo to console as well
                    no_image_count += 1
                    # Continue processing even without a matching image

                # Additional check to make sure it's actually a PAGE XML file
                if not is_page_xml(xml_file, logger):
                    logger.info(f"Skipping {xml_file} - not detected as a PAGE XML file")
                    skipped_count += 1
                    continue

                logger.info(f"Processing PAGE XML file: {xml_file}")

                # Create temporary files in the system temp directory
                temp_file1 = os.path.join(work_temp_dir, f"{os.path.basename(xml_file)}.no_transkribus.xml")
                temp_file2 = os.path.join(work_temp_dir, f"{os.path.basename(xml_file)}.latest.xml")

                # Step 1: Remove TranskribusMetadata
                if not remove_transkribus_metadata(xml_file, temp_file1, logger):
                    logger.error(f"Failed to remove TranskribusMetadata from {xml_file}")
                    error_count += 1
                    continue

                # Step 2: Convert to latest PAGE format
                if not convert_to_latest_page_format(temp_file1, temp_file2, logger):
                    logger.error(f"Failed to convert {xml_file} to latest PAGE format")
                    error_count += 1
                    continue

                # Check if temp_file2 actually exists before copying
                if not os.path.exists(temp_file2):
                    logger.error(f"Expected output file {temp_file2} does not exist")
                    error_count += 1
                    continue

                # Copy the latest PAGE format file to the target directory
                page_target_file = os.path.join(target_dir, os.path.basename(xml_file))
                try:
                    shutil.copy2(temp_file2, page_target_file)
                    logger.info(f"Copied latest PAGE format to {page_target_file}")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to copy processed file to target: {e}")
                    error_count += 1
                    continue

                # Step 3 (Optional): Convert to ALTO
                if convert_to_alto:
                    try:
                        convert_page_to_alto(temp_file2, target_dir, logger)
                    except Exception as e:
                        logger.error(f"Failed to convert to ALTO: {e}")
                        # Don't count this as an error that prevents processing

            except Exception as e:
                logger.error(f"Error processing {xml_file}: {e}")
                print(f"ERROR: Error processing {xml_file}: {e}")
                error_count += 1
                continue

    # Log summary
    logger.info(f"Directory summary for {work_name}:")
    logger.info(f"  - PAGE XML files processed: {processed_count}")
    logger.info(f"  - Files skipped (not PAGE XML): {skipped_count}")
    logger.info(f"  - XML files without matching images: {no_image_count}")
    logger.info(f"  - Files with processing errors: {error_count}")

    # Print summary to console as well
    print(f"Directory summary for {work_name}:")
    print(f"  - PAGE XML files processed: {processed_count}")
    print(f"  - Files skipped (not PAGE XML): {skipped_count}")
    print(f"  - XML files without matching images: {no_image_count}")
    print(f"  - Files with processing errors: {error_count}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Enhanced processing of PAGE XML and image files from work directories")
    parser.add_argument('src_dir', help='Source directory containing work directories')
    parser.add_argument('target_dir', help='Target directory for processed files')
    parser.add_argument('--log', help='Log file path', default='page_processor.log')
    parser.add_argument('--alto', action='store_true', help='Enable conversion to ALTO XML (optional)')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue processing remaining directories if an error occurs in one directory')
    args = parser.parse_args()

    # Ensure the directories exist
    src_dir = Path(args.src_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    log_file = args.log

    if not src_dir.is_dir():
        print(f"Error: Source directory '{src_dir}' does not exist", file=sys.stderr)
        return 1

    os.makedirs(target_dir, exist_ok=True)

    # Set up logging
    logger = setup_logging(log_file)
    logger.info("Starting PAGE XML processing")

    # Get list of work directories
    work_dirs = get_work_directories(src_dir)
    logger.info(f"Found work directories: {work_dirs}")

    # Log processing mode
    if args.alto:
        logger.info("Processing mode: Convert to latest PAGE XML format and ALTO")
    else:
        logger.info("Processing mode: Convert to latest PAGE XML format only")

    # Process each work directory
    error_dirs = []
    for work_name in work_dirs:
        try:
            process_work_directory(src_dir, work_name, target_dir, logger, convert_to_alto=args.alto)
        except Exception as e:
            error_message = f"Error processing directory {work_name}: {e}"
            logger.error(error_message)
            print(f"ERROR: {error_message}")
            error_dirs.append(work_name)

            if not args.continue_on_error:
                logger.error("Stopping processing due to error. Use --continue-on-error to process remaining directories.")
                print("Stopping processing due to error. Use --continue-on-error to process remaining directories.")
                return 1

    # Final summary
    total_dirs = len(work_dirs)
    successful_dirs = total_dirs - len(error_dirs)
    logger.info(f"Processing completed. Successfully processed {successful_dirs}/{total_dirs} directories.")

    if error_dirs:
        logger.warning(f"The following directories had errors: {', '.join(error_dirs)}")
        print(f"WARNING: {len(error_dirs)} directories had errors. See log for details.")
    else:
        logger.info("All directories were processed successfully.")
        print("All directories were processed successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
