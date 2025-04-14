

::
:: Loop through all files within the specified folder
::

FOR %%c in ("input\*.xml") DO (
  java -jar ..\PageConverter.jar -source-xml "%%c" -target-xml "output\%%~nc.xml" -text-filter filter_rules.xml
)