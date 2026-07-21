import assert from 'node:assert/strict';
import test from 'node:test';
import { extractChatTemplate, normalizeLicenses, selectLicense } from '../cloud/materialize-frontier-spec.mjs';

test('license selection is explicit and fail closed', () => {
  assert.deepEqual(normalizeLicenses(['MIT', 'mit', 'Apache-2.0']), ['apache-2.0', 'mit']);
  assert.equal(selectLicense('Apache-2.0', null, 'model'), 'apache-2.0');
  assert.equal(selectLicense(['mit', 'apache-2.0'], 'MIT', 'model'), 'mit');
  assert.throws(() => selectLicense(['mit', 'apache-2.0'], null, 'model'), /multiple licenses/);
  assert.throws(() => selectLicense(null, null, 'model'), /no license/);
});

test('chat template selection refuses ambiguous absence', () => {
  assert.equal(extractChatTemplate({ chat_template: 'direct' }), 'direct');
  assert.equal(extractChatTemplate({ chat_template: { default: 'default' } }), 'default');
  assert.equal(extractChatTemplate({}, 'file-template'), 'file-template');
  assert.throws(() => extractChatTemplate({}), /no unambiguous/);
});
