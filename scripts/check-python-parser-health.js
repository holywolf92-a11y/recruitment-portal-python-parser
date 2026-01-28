/**
 * Python Parser Service Health Check
 * 
 * Checks if the Python parser service is running and accessible
 */

const https = require('https');
const http = require('http');

const PYTHON_PARSER_URL = process.env.PYTHON_CV_PARSER_URL || 
  'https://recruitment-portal-python-parser-production.up.railway.app';

console.log('\n╔════════════════════════════════════════════════════════════════╗');
console.log('║        PYTHON PARSER SERVICE HEALTH CHECK                     ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

console.log(`Python Parser URL: ${PYTHON_PARSER_URL}\n`);

// Check 1: Root endpoint
async function checkRoot() {
  console.log('🔍 Checking root endpoint...');
  return new Promise((resolve) => {
    const url = new URL(PYTHON_PARSER_URL);
    const client = url.protocol === 'https:' ? https : http;
    
    const req = client.get(`${PYTHON_PARSER_URL}/`, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode === 200) {
          try {
            const json = JSON.parse(data);
            console.log('✅ Python parser is responding');
            console.log(`   Service: ${json.service || 'Unknown'}`);
            console.log(`   Version: ${json.version || 'Unknown'}`);
            console.log(`   Status: ${json.status || 'Unknown'}`);
            console.log(`   Environment: ${json.environment || 'Unknown'}`);
            resolve(true);
          } catch (e) {
            console.log('✅ Python parser is responding (non-JSON response)');
            console.log(`   Response: ${data.substring(0, 100)}`);
            resolve(true);
          }
        } else {
          console.log(`❌ Python parser returned status ${res.statusCode}`);
          resolve(false);
        }
      });
    });
    
    req.on('error', (err) => {
      console.log(`❌ Python parser is not accessible: ${err.message}`);
      console.log(`   URL: ${PYTHON_PARSER_URL}`);
      resolve(false);
    });
    
    req.setTimeout(5000, () => {
      req.destroy();
      console.log('❌ Python parser health check timed out');
      resolve(false);
    });
  });
}

// Check 2: Health endpoint
async function checkHealth() {
  console.log('\n🔍 Checking health endpoint...');
  return new Promise((resolve) => {
    const url = new URL(PYTHON_PARSER_URL);
    const client = url.protocol === 'https:' ? https : http;
    
    const req = client.get(`${PYTHON_PARSER_URL}/health`, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode === 200) {
          try {
            const json = JSON.parse(data);
            console.log('✅ Health endpoint is working');
            console.log(`   Status: ${json.status || 'Unknown'}`);
            if (json.version) console.log(`   Version: ${json.version}`);
            resolve(true);
          } catch (e) {
            console.log('✅ Health endpoint is responding');
            console.log(`   Response: ${data.substring(0, 100)}`);
            resolve(true);
          }
        } else {
          console.log(`❌ Health endpoint returned status ${res.statusCode}`);
          resolve(false);
        }
      });
    });
    
    req.on('error', (err) => {
      console.log(`❌ Could not check health endpoint: ${err.message}`);
      resolve(false);
    });
    
    req.setTimeout(5000, () => {
      req.destroy();
      console.log('❌ Health endpoint check timed out');
      resolve(false);
    });
  });
}

// Check 3: Test categorize endpoint (without auth - should return 401)
async function checkCategorizeEndpoint() {
  console.log('\n🔍 Checking categorize-document endpoint...');
  return new Promise((resolve) => {
    const url = new URL(PYTHON_PARSER_URL);
    const client = url.protocol === 'https:' ? https : http;
    
    const testPayload = JSON.stringify({
      file_content: 'test',
      file_name: 'test.pdf',
      mime_type: 'application/pdf'
    });
    
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(testPayload)
      }
    };
    
    const req = client.request(`${PYTHON_PARSER_URL}/categorize-document`, options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        // 401 is expected (missing HMAC signature)
        if (res.statusCode === 401) {
          console.log('✅ Categorize endpoint is accessible (401 = auth required, which is correct)');
          resolve(true);
        } else if (res.statusCode === 400 || res.statusCode === 422) {
          console.log('✅ Categorize endpoint is accessible (validation error, which is expected)');
          resolve(true);
        } else {
          console.log(`⚠️  Categorize endpoint returned status ${res.statusCode}`);
          console.log(`   Response: ${data.substring(0, 200)}`);
          resolve(false);
        }
      });
    });
    
    req.on('error', (err) => {
      console.log(`❌ Could not check categorize endpoint: ${err.message}`);
      resolve(false);
    });
    
    req.write(testPayload);
    req.end();
    
    req.setTimeout(5000, () => {
      req.destroy();
      console.log('❌ Categorize endpoint check timed out');
      resolve(false);
    });
  });
}

// Main diagnostic
async function runDiagnostics() {
  const rootOk = await checkRoot();
  
  if (!rootOk) {
    console.log('\n❌ Python parser service is not accessible. Possible causes:');
    console.log('   1. Service crashed on startup');
    console.log('   2. Missing PYTHON_HMAC_SECRET environment variable');
    console.log('   3. Missing OPENAI_API_KEY environment variable');
    console.log('   4. Port conflict or binding error');
    console.log('   5. Missing Python dependencies');
    console.log('   6. Service not deployed or deployment failed');
    console.log('\n💡 Check Railway logs:');
    console.log('   railway logs --service recruitment-portal-python-parser');
    console.log('\n💡 Check Railway dashboard for:');
    console.log('   - Deployment status');
    console.log('   - Environment variables (PYTHON_HMAC_SECRET, OPENAI_API_KEY)');
    console.log('   - Build logs');
    return;
  }
  
  await checkHealth();
  await checkCategorizeEndpoint();
  
  console.log('\n✅ Python parser service appears to be running normally');
  console.log('\n💡 If you\'re still experiencing issues:');
  console.log('   1. Check Railway deployment logs');
  console.log('   2. Verify environment variables are set:');
  console.log('      - PYTHON_HMAC_SECRET');
  console.log('      - OPENAI_API_KEY');
  console.log('      - SUPABASE_URL (optional)');
  console.log('      - SUPABASE_SERVICE_ROLE_KEY (optional)');
  console.log('   3. Check for recent code changes');
  console.log('   4. Review error logs in Railway dashboard');
}

runDiagnostics().catch(console.error);
