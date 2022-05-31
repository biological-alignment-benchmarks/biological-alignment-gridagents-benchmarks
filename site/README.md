# aintelope site documentation

The aintelope site documentation is written in Markdown and
converted into a static site with 
[Pelican](https://docs.getpelican.com/en/latest/).

The site was initially created with the pelican-quickstart.

# Writing

Write posts (e.g., announcements) in content.
Write documentation in content/pages.

See more about writing content [here](https://docs.getpelican.com/en/latest/content.html).

# Previewing

  make serve

See the result at [http://127.0.0.1:8000].

# Publishing

Publishing to the official site [https://www.aintelope.net] 
requires S3 credentials on the team's AWS account.

  make html
  make s3_upload
