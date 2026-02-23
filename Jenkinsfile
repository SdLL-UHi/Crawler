@Library('web-service-helper-lib') _

pipeline {
    agent { label 'docker' }

    options {
        ansiColor('xterm')
    }

    stages {
        stage('Build Docker Image') {
            agent {
                docker {
                    image 'docker:24'
                    reuseNode true
                    args '-u root:root -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            environment {
                HOME = "${env.WORKSPACE}"
                DOCKER_CONFIG = "${env.WORKSPACE}/.docker"
            }
            steps {
                sh 'mkdir -p "$DOCKER_CONFIG"'
                script {
                    dockerImage = docker.build("ghcr.io/sdll-uhi/crawler:${env.BUILD_NUMBER}")
                    docker.withRegistry('https://ghcr.io', 'ssejenkins-by-elscha') {
                        dockerImage.push()
                        dockerImage.push("latest")
                    }
                }
            }
            post {
                always {
                    sh 'rm -rf "$DOCKER_CONFIG" || true'
                }
            }
        }
    }
}