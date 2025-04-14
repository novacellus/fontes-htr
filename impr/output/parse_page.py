#!/usr/bin/env python3
from kraken.lib import xml
from kraken import serialization
import dataclasses
doc = '/home/krzys/Kod/fontes/repos/GB/opera_nova/BRUDZEW_Comment/page/839309_0027_31120460.xml'
doc_alto = 'alto.xml'
parsed_doc = xml.XMLPage(doc_alto)
container = parsed_doc.to_container()
seg_serialized = serialization.serialize(container,
                                         template='alto')



# Create a list of text records
#texts = [line.text for line in container.lines]

# Add the texts back to the lines in the container
#lines = []
#for i, line in enumerate(container.lines):
    # Create a new line with the text from the original line
#    updated_line = dataclasses.replace(line, text=texts[i])
#    lines.append(updated_line)

# Create a new container with the updated lines
#updated_container = dataclasses.replace(container, lines=lines)


#texts = [line.text for line in container.lines]
#results = dataclasses.replace(_it.bounds, lines=records)
with open('output.xml', 'w') as fp:
    fp.write(seg_serialized)
