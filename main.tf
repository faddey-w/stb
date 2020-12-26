terraform {
  required_version = ">= 0.13"
  backend "s3" {
    bucket = "stb-infra"
    key    = "terraform/state.json"
    region = "us-west-2"
    profile = "stb-ops"
  }
  required_providers {
    null = {
      source  = "hashicorp/null"
      version = "2.1.2"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "3.14.1"
    }
  }
}

provider "aws" {
  region = "us-west-2"
  profile = "stb-ops"
}


resource "aws_s3_bucket" "replaystorage" {
  bucket = "stb-replaystorage"
  acl    = "public-read"
  policy = <<-POLICY
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": [
                    "s3:GetObject"
                ],
                "Resource": [
                    "arn:aws:s3:::stb-replaystorage/*"
                ]
            }
        ]
    }
POLICY
}



resource "aws_cloudfront_distribution" "s3_distribution" {
  origin {
    domain_name = aws_s3_bucket.replaystorage.bucket_regional_domain_name
    origin_id   = "s3-replaystorage"
  }

  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "replay-list.json"

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "s3-replaystorage"

    forwarded_values {
      query_string = false

      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "allow-all"
    min_ttl                = 0
    default_ttl            = 600
    max_ttl                = 86400
  }


  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

output "cloudfront_domain" {
  value = aws_cloudfront_distribution.s3_distribution.domain_name
}
