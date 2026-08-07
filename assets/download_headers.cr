require "file/tempfile"
require "http/client"
require "yaml"

FILES = {
  "LICENSE"        => "LICENSE",
  "llama.h"        => "include/llama.h",
  "ggml.h"         => "ggml/include/ggml.h",
  "ggml-cpu.h"     => "ggml/include/ggml-cpu.h",
  "ggml-alloc.h"   => "ggml/include/ggml-alloc.h",
  "ggml-backend.h" => "ggml/include/ggml-backend.h",
  "ggml-opt.h"     => "ggml/include/ggml-opt.h",
  "gguf.h"         => "ggml/include/gguf.h",
}

assets_dir = __DIR__
project_dir = File.expand_path("..", assets_dir)
shard_path = File.join(project_dir, "shard.yml")
current_version = YAML.parse(File.read(shard_path))["version"].as_s
current_match = current_version.match(/\A0\.(\d+)\.\d+\z/)

unless current_match
  STDERR.puts "Invalid shard version format: #{current_version} (expected 0.<build>.<patch>)"
  exit 1
end

if ARGV.size > 1
  STDERR.puts "Usage: crystal run assets/download_headers.cr [build]"
  exit 1
end

requested_build = ARGV.first?.try(&.sub(/\Ab/, ""))
unless requested_build.nil? || requested_build.matches?(/\A\d+\z/)
  STDERR.puts "Invalid build: #{ARGV.first} (expected digits or b<digits>)"
  exit 1
end

build = requested_build || current_match[1]
target_version = "0.#{build}.0"
llama_build = "b#{build}"
base_url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/#{llama_build}"
replacements = [] of Tuple(String, String)

puts "Downloading llama.cpp headers version #{llama_build}..."

begin
  FILES.each do |filename, source_path|
    url = "#{base_url}/#{source_path}"
    tempfile = File.tempfile("llama-header", ".tmp", dir: assets_dir)
    replacements << {File.join(assets_dir, filename), tempfile.path}

    begin
      HTTP::Client.get(url) do |response|
        unless response.status.success?
          raise IO::Error.new("Failed to download #{url}: HTTP #{response.status_code}")
        end

        IO.copy(response.body_io, tempfile)
      end
    ensure
      tempfile.close
    end

    puts "Downloaded #{filename}"
  end

  if requested_build
    shard_contents = File.read(shard_path).sub(/^version:.*$/, "version: #{target_version}")
    tempfile = File.tempfile("shard", ".yml.tmp", dir: project_dir)
    tempfile.print(shard_contents)
    tempfile.close
    replacements << {shard_path, tempfile.path}
  end

  replacements.each do |destination, temporary_path|
    File.rename(temporary_path, destination)
  end
ensure
  replacements.each do |_destination, temporary_path|
    File.delete?(temporary_path)
  end
end

puts "Downloaded llama.cpp headers from version #{llama_build}"
if requested_build
  puts "Updated shard version to #{target_version}"
  Process.run("git", ["diff", "--stat", "--", shard_path, assets_dir], output: STDOUT, error: STDERR)
end
