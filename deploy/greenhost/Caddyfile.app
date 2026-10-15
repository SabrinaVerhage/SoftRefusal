softrefusal.sabrinaverhage.com {
	encode zstd gzip
	header {
		X-Content-Type-Options nosniff
		Referrer-Policy no-referrer
		X-Robots-Tag "noindex, nofollow"
		-Server
	}
	request_body {
		max_size 20KB
	}
	respond /internal/* 404
	reverse_proxy 127.0.0.1:8780
}
